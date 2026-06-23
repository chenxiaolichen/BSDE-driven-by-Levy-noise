import os
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
    N = 40
    dt = T / N
    dim = 4

    batch_size = 32  
    accum_steps = 32  
    update_steps = 8000
    epochs = update_steps * accum_steps

    pretrain_steps = 5000  
    lr = 2e-4

    hidden = 256
    n_layers = 5
    alpha1 = 1.5
    alpha2 = 1.2

    levy_R = 8.0
    levy_M = 200
    levy_eps = 0.05

    x0_low = -math.pi
    x0_high = math.pi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

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


def C_4D_alpha(alpha: float) -> float:
    num = alpha * (2.0 ** (alpha - 1.0)) * gamma_func((alpha + 4.0) / 2.0)
    den = (math.pi ** 2) * gamma_func(1.0 - alpha / 2.0)
    return num / den


class FractionalCoupledSineGordon4D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.C1, self.C2 = float(C_4D_alpha(cfg.alpha1)), float(C_4D_alpha(cfg.alpha2))
        r = cfg.levy_eps

        var_1 = self.C1 * (math.pi ** 2) / (2.0 * (2.0 - cfg.alpha1)) * (r ** (2.0 - cfg.alpha1))
        var_2 = self.C2 * (math.pi ** 2) / (2.0 * (2.0 - cfg.alpha2)) * (r ** (2.0 - cfg.alpha2))
        self.sigma_u, self.sigma_v = math.sqrt(var_1), math.sqrt(var_2)

    def g1_terminal(self, x):
        return torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * torch.cos(2.0 * x[:, 2:3]) * torch.sin(
            2.0 * x[:, 3:4]) * math.sin(self.cfg.T)

    def g2_terminal(self, x):
        return torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * torch.sin(2.0 * x[:, 2:3]) * torch.cos(
            2.0 * x[:, 3:4]) * math.cos(self.cfg.T)

    def u_exact(self, t, x):
        return torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * torch.cos(2.0 * x[:, 2:3]) * torch.sin(
            2.0 * x[:, 3:4]) * torch.sin(t)

    def v_exact(self, t, x):
        return torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * torch.sin(2.0 * x[:, 2:3]) * torch.cos(
            2.0 * x[:, 3:4]) * torch.cos(t)

    def mc_fractional_integral_4d(self, net, t, x, alpha, C_alpha, var_idx):
        device, batch = x.device, x.shape[0]
        M = self.cfg.levy_M
        r, R = float(self.cfg.levy_eps), float(self.cfg.levy_R)

        u_rho = torch.linspace(0.5 / M, 1.0 - 0.5 / M, M, device=device).view(1, M)
        rho = (r ** (-alpha) - u_rho * (r ** (-alpha) - R ** (-alpha))).pow(-1.0 / alpha)

        dirs = torch.randn(batch, M, 4, device=device)
        dirs = dirs / torch.norm(dirs, dim=2, keepdim=True)
        y_vec = rho.unsqueeze(-1) * dirs

        x_exp = x.unsqueeze(1)
        x_plus_y = (x_exp + y_vec).view(-1, 4)
        x_minus_y = (x_exp - y_vec).view(-1, 4)
        t_exp = t.unsqueeze(1).expand(-1, M, -1).reshape(-1, 1)

        val_plus = net(t_exp, x_plus_y)[var_idx].view(batch, M)
        val_minus = net(t_exp, x_minus_y)[var_idx].view(batch, M)
        val_center = net(t, x)[var_idx].view(batch, 1)

        diff = val_plus + val_minus - 2.0 * val_center
        K = C_alpha * (math.pi ** 2 / alpha) * (r ** (-alpha) - R ** (-alpha))
        return K * diff.mean(dim=1, keepdim=True)

    def f1(self, net, t, x, v_at_x):
        ue, ve = self.u_exact(t, x), self.v_exact(t, x)
        ute = torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * torch.cos(2.0 * x[:, 2:3]) * torch.sin(
            2.0 * x[:, 3:4]) * torch.cos(t)
        integral_u = self.mc_fractional_integral_4d(net, t, x, self.cfg.alpha1, self.C1, var_idx=0)
        eigen_val = 4.0 ** self.cfg.alpha1
        return integral_u + torch.sin(v_at_x) + (eigen_val * ue - ute - torch.sin(ve))

    def f2(self, net, t, x, u_at_x):
        ue, ve = self.u_exact(t, x), self.v_exact(t, x)
        vte = -torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * torch.sin(2.0 * x[:, 2:3]) * torch.cos(
            2.0 * x[:, 3:4]) * torch.sin(t)
        integral_v = self.mc_fractional_integral_4d(net, t, x, self.cfg.alpha2, self.C2, var_idx=1)
        eigen_val = 4.0 ** self.cfg.alpha2
        return integral_v + torch.sin(u_at_x) + (eigen_val * ve - vte - torch.sin(ue))


# ==========================================
# 3. Solver 
# ==========================================
class ExplicitHarmonicNet4D(nn.Module):
    def __init__(self, hidden=256, n_layers=5):
        super().__init__()
        in_dim = 1 + 4 + 4 + 4 + 4 + 4 + 1 + 1
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())

        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh()))

        self.head_u = nn.Linear(hidden, 1)
        self.head_v = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, t, x):
        if t.dim() == 1: t = t.view(-1, 1)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        sin_2x = torch.sin(2.0 * x)
        cos_2x = torch.cos(2.0 * x)
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)

        features = torch.cat([t, x, sin_x, cos_x, sin_2x, cos_2x, sin_t, cos_t], dim=-1)

        H = self.encoder(features)
        for layer in self.hidden_layers:
            H = H + layer(H)
        return self.head_u(H), self.head_v(H)


class DeepBSDESolver4D:
    def __init__(self, cfg, equation, device):
        self.cfg, self.eq, self.device = cfg, equation, device
        self.net = ExplicitHarmonicNet4D(hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2000, gamma=0.9)

    def train_epoch(self, current_step):
        batch, device, cfg, eq = self.cfg.batch_size, self.device, self.cfg, self.eq
        dt = cfg.dt

        curriculum_ratio = min(1.0, current_step / (cfg.epochs * 0.5))
        min_start_step = int((cfg.N - 1) * (1.0 - curriculum_ratio))
        n_start = random.randint(min_start_step, cfg.N - 1)

        t = torch.full((batch, 1), n_start * dt, device=device)
        x_u = torch.empty(batch, 4, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        x_v = torch.empty(batch, 4, device=device).uniform_(cfg.x0_low, cfg.x0_high)

        Y_u, _ = self.net(t, x_u)
        _, Y_v = self.net(t, x_v)

        for step in range(n_start, cfg.N):
            x_u_g = x_u.clone().detach().requires_grad_(True)
            x_v_g = x_v.clone().detach().requires_grad_(True)

            u_val_at_xu, v_val_at_xu = self.net(t, x_u_g)
            u_val_at_xv, v_val_at_xv = self.net(t, x_v_g)

            Z_u = torch.autograd.grad(u_val_at_xu.sum(), x_u_g, create_graph=True)[0]
            Z_v = torch.autograd.grad(v_val_at_xv.sum(), x_v_g, create_graph=True)[0]

            dW_u = torch.randn(batch, 4, device=device) * math.sqrt(dt)
            dW_v = torch.randn(batch, 4, device=device) * math.sqrt(dt)

            Y_u = Y_u - eq.f1(self.net, t, x_u_g, v_val_at_xu) * dt + torch.sum(Z_u * dW_u, dim=1,
                                                                                keepdim=True) * eq.sigma_u
            Y_v = Y_v - eq.f2(self.net, t, x_v_g, u_val_at_xv) * dt + torch.sum(Z_v * dW_v, dim=1,
                                                                                keepdim=True) * eq.sigma_v

            x_u = x_u_g.detach() + eq.sigma_u * dW_u
            x_v = x_v_g.detach() + eq.sigma_v * dW_v
            t = t + dt

        u_term = eq.g1_terminal(x_u)
        v_term = eq.g2_terminal(x_v)
        path_loss = nn.MSELoss()(Y_u, u_term) + nn.MSELoss()(Y_v, v_term)

        x_anchor = torch.empty(batch, 4, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        t_anchor = torch.full((batch, 1), cfg.T, device=device)
        u_anchor, v_anchor = self.net(t_anchor, x_anchor)
        anchor_loss = nn.MSELoss()(u_anchor, eq.g1_terminal(x_anchor)) + nn.MSELoss()(v_anchor,
                                                                                      eq.g2_terminal(x_anchor))

        return path_loss + 10.0 * anchor_loss

    def train(self):
        print(f"启动 4D Sine-Gordon 预热...")
        pre_opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for s in range(1, self.cfg.pretrain_steps + 1):
            pre_opt.zero_grad()
            t_rand = torch.rand(4096, 1, device=self.device) * self.cfg.T
            x_rand = torch.empty(4096, 4, device=self.device).uniform_(self.cfg.x0_low, self.cfg.x0_high)

            u_pred, v_pred = self.net(t_rand, x_rand)
            loss = nn.MSELoss()(u_pred, self.eq.g1_terminal(x_rand)) + nn.MSELoss()(v_pred, self.eq.g2_terminal(x_rand))
            loss.backward();
            pre_opt.step()

            if s % 1000 == 0: print(f'[Legal Warm-up] step {s:05d} loss={loss.item():.5e}')

        print(f"\n 开始 4D 主阶段训练 | Accum Steps: {self.cfg.accum_steps})...")
        self.opt.zero_grad()
        for it in range(1, self.cfg.epochs + 1):
            loss_scaled = self.train_epoch(it) / self.cfg.accum_steps
            loss_scaled.backward()

            if it % self.cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
                self.opt.step();
                self.opt.zero_grad();
                self.scheduler.step()

                real_step = it // self.cfg.accum_steps
                if real_step % 100 == 0:
                    print(
                        f"Update Step {real_step:05d}/{self.cfg.update_steps}  Loss={loss_scaled.item() * self.cfg.accum_steps:.5e}  LR={self.opt.param_groups[0]['lr']:.1e}")


# ==========================================
# 4. 
# ==========================================
if __name__ == "__main__":
    eq = FractionalCoupledSineGordon4D(cfg)
    solver = DeepBSDESolver4D(cfg, eq, cfg.device)

    t0 = time.time()
    solver.train()
    print(f"\n 4D 训练完毕！总耗时: {time.time() - t0:.1f}s")

    os.makedirs('checkpoints', exist_ok=True)
    weight_path = 'checkpoints/sine_gordon_net_4d.pth'
    torch.save(solver.net.state_dict(), weight_path)
    print(f" 权重已成功保存至 {weight_path}")

    solver.net.eval()

    print("\n 正在计算 4D 相对 L2 误差")
    N_global = 100000
    with torch.no_grad():
        t_global = torch.rand(N_global, 1, device=cfg.device) * cfg.T
        x_global = torch.empty(N_global, 4, device=cfg.device).uniform_(cfg.x0_low, cfg.x0_high)

        u_pred_g, v_pred_g = solver.net(t_global, x_global)
        u_exact_g = eq.u_exact(t_global, x_global)
        v_exact_g = eq.v_exact(t_global, x_global)

        global_L2_u = torch.sqrt(torch.mean((u_pred_g - u_exact_g) ** 2)) / torch.sqrt(torch.mean(u_exact_g ** 2))
        global_L2_v = torch.sqrt(torch.mean((v_pred_g - v_exact_g) ** 2)) / torch.sqrt(torch.mean(v_exact_g ** 2))

    print("=" * 50)
    print(f' 4D 相对 L2 误差 - U (Global): {global_L2_u.item():.6e}')
    print(f' 4D 相对 L2 误差 - V (Global): {global_L2_v.item():.6e}')
    print("=" * 50 + "\n")

    time_steps = [0.0, 0.5, 1.0]
    res = 60
    x_lin = np.linspace(-math.pi, math.pi, res).astype(np.float32)
    X1, X2 = np.meshgrid(x_lin, x_lin)

    x3_fixed = np.zeros_like(X1.ravel())
    x4_fixed = np.full_like(X1.ravel(), math.pi / 4.0)
    x_flat_slice = torch.tensor(np.stack([X1.ravel(), X2.ravel(), x3_fixed, x4_fixed], axis=1), device=cfg.device)


    def evaluate_and_plot_variable_4d(var_name):
        print(f" {var_name.upper()} 的 3x3 4D切片热力图...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        plt.rcParams.update({'font.size': 12})

        for i, t_val in enumerate(time_steps):
            t_eval = torch.full((res * res, 1), float(t_val), device=cfg.device)
            with torch.no_grad():
                pred_u, pred_v = solver.net(t_eval, x_flat_slice)
                if var_name == 'u':
                    pred = pred_u.cpu().numpy().reshape(res, res)
                    exact = eq.u_exact(t_eval, x_flat_slice).cpu().numpy().reshape(res, res)
                else:
                    pred = pred_v.cpu().numpy().reshape(res, res)
                    exact = eq.v_exact(t_eval, x_flat_slice).cpu().numpy().reshape(res, res)

            abs_err = np.abs(exact - pred)
            norm_exact = np.sqrt(np.mean(exact ** 2))

            if norm_exact < 1e-7:
                L2_err = np.sqrt(np.mean((pred - exact) ** 2))
                err_title = f"Abs Error {var_name.upper()} (Abs L2: {L2_err:.4f})"
            else:
                L2_err = np.sqrt(np.mean((pred - exact) ** 2)) / norm_exact
                err_title = f"Abs Error {var_name.upper()} (Rel L2: {L2_err:.4f})"

            ax1 = axes[i, 0]
            im1 = ax1.pcolormesh(X1, X2, exact, cmap='jet', shading='auto')
            ax1.set_title(f"Exact {var_name.upper()} (t={t_val}, slice)", fontsize=16)
            ax1.set_xlabel("$x_1$");
            ax1.set_ylabel("$x_2$")
            fig.colorbar(im1, ax=ax1)

            ax2 = axes[i, 1]
            im2 = ax2.pcolormesh(X1, X2, pred, cmap='jet', shading='auto')
            ax2.set_title(f"Predicted {var_name.upper()} (t={t_val})", fontsize=16)
            ax2.set_xlabel("$x_1$");
            ax2.set_ylabel("$x_2$")
            fig.colorbar(im2, ax=ax2)

            ax3 = axes[i, 2]
            im3 = ax3.pcolormesh(X1, X2, abs_err, cmap='magma', shading='auto')
            ax3.set_title(err_title, fontsize=16)
            ax3.set_xlabel("$x_1$");
            ax3.set_ylabel("$x_2$")
            fig.colorbar(im3, ax=ax3)

        plt.tight_layout()
        save_name = f'sine_gordon_4d_results_{var_name.upper()}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"🎉 {var_name.upper()} 高清热力图已保存: {save_name}")


    evaluate_and_plot_variable_4d('u')
    evaluate_and_plot_variable_4d('v')

    print("\n 正在计算 4D 误差实时演化曲线...")
    num_time_steps = 101
    time_array = np.linspace(0.0, cfg.T, num_time_steps)
    err_u_abs, err_v_abs, err_u_rel, err_v_rel = [], [], [], []

    n_curve_samples = 10000  

    with torch.no_grad():
        for t_val in time_array:
            t_eval = torch.full((n_curve_samples, 1), float(t_val), device=cfg.device)
            x_eval = torch.empty(n_curve_samples, 4, device=cfg.device).uniform_(cfg.x0_low, cfg.x0_high)

            up, vp = solver.net(t_eval, x_eval)
            ue = eq.u_exact(t_eval, x_eval)
            ve = eq.v_exact(t_eval, x_eval)

            up, vp = up.cpu().numpy(), vp.cpu().numpy()
            ue, ve = ue.cpu().numpy(), ve.cpu().numpy()

            abs_u = np.sqrt(np.mean((up - ue) ** 2))
            abs_v = np.sqrt(np.mean((vp - ve) ** 2))
            err_u_abs.append(abs_u)
            err_v_abs.append(abs_v)

            norm_ue = np.sqrt(np.mean(ue ** 2))
            err_u_rel.append(abs_u / norm_ue if norm_ue > 1e-7 else abs_u)

            norm_ve = np.sqrt(np.mean(ve ** 2))
            err_v_rel.append(abs_v / norm_ve if norm_ve > 1e-7 else abs_v)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time_array, err_u_abs, 'b-', linewidth=2.5, label='Absolute Error - U')
    plt.plot(time_array, err_v_abs, 'r--', linewidth=2.5, label='Absolute Error - V')
    plt.title('Evolution of Absolute $L^2$ Error (4D)', fontsize=14, fontweight='bold')
    plt.xlabel('Time ($t$)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.xlim(1.0, 0.0);
    plt.grid(True, linestyle='--', alpha=0.7);
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time_array, err_u_rel, 'b-', linewidth=2.5, label='Relative Error - U')
    plt.plot(time_array, err_v_rel, 'r--', linewidth=2.5, label='Relative Error - V')
    plt.title('Evolution of Relative $L^2$ Error (4D)', fontsize=14, fontweight='bold')
    plt.xlabel('Time ($t$)', fontsize=12)
    plt.ylabel('Relative Error', fontsize=12)
    plt.xlim(1.0, 0.0);
    plt.grid(True, linestyle='--', alpha=0.7);
    plt.legend()

    plt.tight_layout()
    plt.savefig('error_evolution_4d.png', dpi=300, bbox_inches='tight')
    print(" 4D 误差实时演化折线图已保存: error_evolution_4d.png")
