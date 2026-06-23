import math
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Config 
# ==========================================
class Config:
    seed = 2026

    T = 1.0
    N = 100
    dt = T / N

    batch_size = 64
    accumulation_steps = 4

    update_steps = 8000
    epochs = update_steps * accumulation_steps
    pretrain_steps = 2000

    lr = 2e-4

    hidden = 128
    n_layers = 4
    alpha1 = 1.5
    alpha2 = 1.2

    levy_R = 8
    levy_M = 200
    levy_eps = 0.05

    x0_low = -12.0
    x0_high = 12.0

    x_min = -4.0
    x_max = 4.0
    x_pts = 201

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()

# ==========================================
# 2. Equation 
# ==========================================
try:
    from scipy.special import gamma as _gamma


    def gamma_func(x):
        return float(_gamma(x))
except Exception:
    def gamma_func(x):
        return float(math.exp(math.lgamma(x)))


def C_alpha(alpha: float) -> float:
    return alpha * gamma_func((1.0 + alpha) / 2.0) / (
                2.0 ** (1.0 - alpha) * math.sqrt(math.pi) * gamma_func(1.0 - alpha / 2.0))


class FractionalCoupledAllenCahn:
    def __init__(self, cfg):
        self.cfg = cfg
        self.C1 = float(C_alpha(cfg.alpha1))
        self.C2 = float(C_alpha(cfg.alpha2))

        r = cfg.levy_eps
        laplace_coeff1 = self.C1 * (r ** (2 - cfg.alpha1)) / (2 - cfg.alpha1)
        laplace_coeff2 = self.C2 * (r ** (2 - cfg.alpha2)) / (2 - cfg.alpha2)

        self.sigma_u = math.sqrt(2.0 * laplace_coeff1)
        self.sigma_v = math.sqrt(2.0 * laplace_coeff2)

    def g1_terminal(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(x_tensor)

    def g2_terminal(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cos(x_tensor)

    def u_exact(self, t_tensor: torch.Tensor, x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(x_tensor) * torch.exp(1.0 - t_tensor)

    def v_exact(self, t_tensor: torch.Tensor, x_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cos(x_tensor) * torch.exp(1.0 - t_tensor)

    def mc_fractional_integral(self, net_callable, t_tensor, x_tensor, alpha):
        device = x_tensor.device
        batch = x_tensor.shape[0]
        M = int(self.cfg.levy_M)
        R = float(self.cfg.levy_R)
        r = float(self.cfg.levy_eps)

        if M % 2 != 0: M += 1
        half_M = M // 2

        u_grid = torch.linspace(0.5 / half_M, 1.0 - 0.5 / half_M, half_M, device=device).view(1, half_M)

        A = r ** (-alpha)
        B = R ** (-alpha)
        s = (A - u_grid * (A - B)).pow(-1.0 / alpha)
        ys = torch.cat([s, -s], dim=1)

        t_flat = t_tensor.expand(-1, M).contiguous().view(-1, 1)
        x_base = x_tensor.expand(-1, M).contiguous()
        x_with_y = (x_base + ys).view(-1, 1)

        vals = net_callable(t_flat, x_with_y).view(batch, M)
        net_x = net_callable(t_tensor, x_tensor).view(batch, 1)

        diff = vals - net_x
        K = 2.0 * (r ** (-alpha) - R ** (-alpha)) / alpha

        return K * diff.mean(dim=1, keepdim=True)

    def f1(self, net_u, net_v, t, x):
        u = net_u(t, x)
        v = net_v(t, x)
        u_ex = self.u_exact(t, x)
        v_ex = self.v_exact(t, x)

        integral_tail = self.mc_fractional_integral(net_u, t, x, self.cfg.alpha1)
        F_u = u - u.pow(3) + v + u_ex.pow(3) + u_ex - v_ex
        return self.C1 * integral_tail + F_u

    def f2(self, net_u, net_v, t, x):
        u = net_u(t, x)
        v = net_v(t, x)
        u_ex = self.u_exact(t, x)
        v_ex = self.v_exact(t, x)

        integral_tail = self.mc_fractional_integral(net_v, t, x, self.cfg.alpha2)
        F_v = v - v.pow(3) - u + v_ex.pow(3) + v_ex + u_ex
        return self.C2 * integral_tail + F_v


# ==========================================
# 3. Solver 
# ==========================================
class FourierFeatureNet(nn.Module):
    def __init__(self, in_dim=2, hidden=128, n_layers=4, sigma=2.0):
        super().__init__()
        self.mapping_size = 64
        self.B = nn.Parameter(torch.randn(in_dim, self.mapping_size) * sigma, requires_grad=False)

        input_dim = 2 * self.mapping_size

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, t, x):
        if t.dim() == 1: t = t.view(-1, 1)
        if x.dim() == 1: x = x.view(-1, 1)
        if x.dim() == 2 and x.shape[1] > 1: x = x[:, :1]

        inp = torch.cat([t, x], dim=1)
        proj = (2.0 * math.pi * inp) @ self.B
        inp_fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(inp_fourier)


class DeepBSDESolver:
    def __init__(self, cfg, equation, device):
        self.cfg = cfg
        self.eq = equation
        self.device = device

        self.net_u = FourierFeatureNet(hidden=cfg.hidden, n_layers=cfg.n_layers, sigma=2.0).to(device)
        self.net_v = FourierFeatureNet(hidden=cfg.hidden, n_layers=cfg.n_layers, sigma=2.0).to(device)

        self.opt = torch.optim.Adam(list(self.net_u.parameters()) + list(self.net_v.parameters()), lr=cfg.lr)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2000, gamma=0.5)
        self.criterion = nn.MSELoss()

    def compute_Z(self, net, t, x):
        x_req = x.clone().detach().requires_grad_(True)
        t_req = t.clone().detach()
        y = net(t_req, x_req)
        grads = torch.autograd.grad(outputs=y.sum(), inputs=x_req, create_graph=True)[0]
        return grads

    def pretrain_supervised(self, steps):
        print(f"Starting Linearized Exponential Warm-up ({steps} steps)...")
        loss_fn = nn.MSELoss()
        pre_opt = torch.optim.Adam(list(self.net_u.parameters()) + list(self.net_v.parameters()), lr=1e-3)
        pretrain_batch = 4096

        for s in range(1, steps + 1):
            pre_opt.zero_grad()

            t_rand = torch.rand(pretrain_batch, 1, device=self.device) * self.cfg.T
            x_rand = torch.empty(pretrain_batch, 1, device=self.device).uniform_(-18.0, 18.0)

            growth_factor = torch.exp(self.cfg.T - t_rand)
            u_tgt = self.eq.g1_terminal(x_rand) * growth_factor
            v_tgt = self.eq.g2_terminal(x_rand) * growth_factor

            loss = loss_fn(self.net_u(t_rand, x_rand), u_tgt) + loss_fn(self.net_v(t_rand, x_rand), v_tgt)
            loss.backward()
            pre_opt.step()

            if s % 500 == 0:
                print(f'[Asymptotic Warm-up] step {s:04d} loss={loss.item():.5e}')

    def train_epoch(self):
        cfg = self.cfg
        device = self.device
        batch = cfg.batch_size

        x_u = torch.empty(batch, 1, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        x_v = torch.empty(batch, 1, device=device).uniform_(cfg.x0_low, cfg.x0_high)

        t = torch.zeros(batch, 1, device=device)

        Y_u = self.net_u(t, x_u)
        Y_v = self.net_v(t, x_v)

        for _ in range(cfg.N):
            dt = cfg.dt
            dW_u = torch.randn(batch, 1, device=device) * math.sqrt(dt)
            dW_v = torch.randn(batch, 1, device=device) * math.sqrt(dt)

            Z_u = self.compute_Z(self.net_u, t, x_u)
            Z_v = self.compute_Z(self.net_v, t, x_v)

            f1 = self.eq.f1(self.net_u, self.net_v, t, x_u)
            f2 = self.eq.f2(self.net_u, self.net_v, t, x_v)

            Y_u = Y_u - f1 * dt + Z_u * self.eq.sigma_u * dW_u
            Y_v = Y_v - f2 * dt + Z_v * self.eq.sigma_v * dW_v

            t = t + dt
            x_u = x_u + self.eq.sigma_u * dW_u
            x_v = x_v + self.eq.sigma_v * dW_v

        u_term = self.eq.g1_terminal(x_u)
        v_term = self.eq.g2_terminal(x_v)
        path_loss = self.criterion(Y_u, u_term) + self.criterion(Y_v, v_term)

        x_reg = torch.empty(batch, 1, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        t_reg = torch.full((batch, 1), cfg.T, device=device)
        u_reg_net = self.net_u(t_reg, x_reg)
        v_reg_net = self.net_v(t_reg, x_reg)

        u_reg_exact = self.eq.g1_terminal(x_reg)
        v_reg_exact = self.eq.g2_terminal(x_reg)
        reg_loss = self.criterion(u_reg_net, u_reg_exact) + self.criterion(v_reg_net, v_reg_exact)

        return path_loss + 10.0 * reg_loss

    def train(self, epochs, log_every=200):
        self.pretrain_supervised(self.cfg.pretrain_steps)
        loss_hist = []
        accum_steps = self.cfg.accumulation_steps
        self.opt.zero_grad()

        print(f"\n 开始 1D Allen-Cahn 主阶段训练  | Accum Steps: {accum_steps})...")
        for it in range(1, epochs + 1):
            loss = self.train_epoch() / accum_steps
            loss.backward()

            if it % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(self.net_u.parameters()) + list(self.net_v.parameters()),
                                               max_norm=5.0)
                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()

                loss_val = loss.item() * accum_steps
                loss_hist.append(loss_val)

                real_step = it // accum_steps
                if real_step % log_every == 0 or real_step == 1:
                    lr_curr = self.opt.param_groups[0]['lr']
                    print(f"Update Step {real_step:04d}/{self.cfg.update_steps}  Loss={loss_val:.5e}  LR={lr_curr:.1e}")

        return loss_hist


# ==========================================
# 4. Train 执行脚本
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(" Fractional Coupled Allen-Cahn Deep BSDE...")
    print(f"Device: {device} | Physical Batch: {cfg.batch_size} | Accumulation: {cfg.accumulation_steps}x")

    eq = FractionalCoupledAllenCahn(cfg)
    solver = DeepBSDESolver(cfg, eq, device)

    t0 = time.time()
    loss_history = solver.train(cfg.epochs, log_every=200)
    t1 = time.time()
    print(f"\n Training finished in {t1 - t0:.1f}s")

    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.x_pts).reshape(-1, 1).astype(np.float32)
    t0_grid = np.zeros_like(x_grid).astype(np.float32)

    t_tensor = torch.tensor(t0_grid, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = solver.net_u(t_tensor, x_tensor).cpu().numpy().flatten()
        v_pred = solver.net_v(t_tensor, x_tensor).cpu().numpy().flatten()

        u_exact = eq.u_exact(t_tensor, x_tensor).cpu().numpy().flatten()
        v_exact = eq.v_exact(t_tensor, x_tensor).cpu().numpy().flatten()

    L2_u = np.sqrt(np.mean((u_pred - u_exact) ** 2)) / np.sqrt(np.mean(u_exact ** 2))
    L2_v = np.sqrt(np.mean((v_pred - v_exact) ** 2)) / np.sqrt(np.mean(v_exact ** 2))

    print("=" * 50)
    print(f' 相对 L2 误差 - u (t=0.0): {L2_u:.6e}')
    print(f' 相对 L2 误差 - v (t=0.0): {L2_v:.6e}')
    print("=" * 50)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_grid, u_exact, label='u exact', linewidth=2)
    plt.plot(x_grid, u_pred, '--', label='u pred', linewidth=2)
    plt.title(f'Allen-Cahn u(t=0, x)\nRel L2={L2_u:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_grid, v_exact, label='v exact', linewidth=2)
    plt.plot(x_grid, v_pred, '--', label='v pred', linewidth=2)
    plt.title(f'Allen-Cahn v(t=0, x)\nRel L2={L2_v:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('allen_cahn_uv_comparison_final.png', dpi=200)
    plt.show()
