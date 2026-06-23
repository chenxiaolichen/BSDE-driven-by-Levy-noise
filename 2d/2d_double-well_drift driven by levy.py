import os
import math
import time
import random
import copy
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
    N = 50
    dt = T / N
    dim = 2

    batch_size = 32
    accum_steps = 16
    update_steps = 12000
    epochs = update_steps * accum_steps

    pretrain_steps = 2000
    lr = 2e-4

    hidden = 128
    n_layers = 4
    alpha1 = 1.5
    alpha2 = 1.2

    levy_R = 8.0
    levy_M_rho = 30
    levy_M_theta = 20
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


def C_2D_alpha(alpha: float) -> float:
    return (alpha * (2.0 ** (alpha - 1.0)) * gamma_func((alpha + 2.0) / 2.0)) / (
            math.pi * gamma_func(1.0 - alpha / 2.0))


class FractionalCoupledComplete2D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.C1, self.C2 = float(C_2D_alpha(cfg.alpha1)), float(C_2D_alpha(cfg.alpha2))
        r = cfg.levy_eps

        var_1 = self.C1 * math.pi * (r ** (2 - cfg.alpha1)) / (2 - cfg.alpha1)
        var_2 = self.C2 * math.pi * (r ** (2 - cfg.alpha2)) / (2 - cfg.alpha2)

        self.sigma_u = math.sqrt(1.0 + var_1)
        self.sigma_v = math.sqrt(1.0 + var_2)

        self.K_u = self.C1 * ((r ** (-cfg.alpha1) - cfg.levy_R ** (-cfg.alpha1)) / cfg.alpha1) * math.pi
        self.K_v = self.C2 * ((r ** (-cfg.alpha2) - cfg.levy_R ** (-cfg.alpha2)) / cfg.alpha2) * math.pi
        self.M_total = cfg.levy_M_rho * cfg.levy_M_theta

        self.y_vec_u = self._generate_y_vec(cfg.alpha1).to(cfg.device)
        self.y_vec_v = self._generate_y_vec(cfg.alpha2).to(cfg.device)

    def _generate_y_vec(self, alpha):
        M_rho, M_theta = self.cfg.levy_M_rho, self.cfg.levy_M_theta
        r, R = float(self.cfg.levy_eps), float(self.cfg.levy_R)
        u_rho = torch.linspace(0.5 / M_rho, 1.0 - 0.5 / M_rho, M_rho)
        rho = (r ** (-alpha) - u_rho * (r ** (-alpha) - R ** (-alpha))).pow(-1.0 / alpha)
        theta = torch.linspace(0.5 / M_theta, 1.0 - 0.5 / M_theta, M_theta) * math.pi
        rho_grid, theta_grid = torch.meshgrid(rho, theta, indexing='ij')
        return torch.stack([rho_grid.reshape(-1) * torch.cos(theta_grid.reshape(-1)),
                            rho_grid.reshape(-1) * torch.sin(theta_grid.reshape(-1))], dim=1)

    def mu(self, x):
        return torch.cat([x[:, 0:1] - x[:, 0:1] ** 3, x[:, 1:2] - x[:, 1:2] ** 3], dim=1)

    def u_exact(self, t, x):
        env = torch.exp(-(x[:, 0:1] ** 2 + x[:, 1:2] ** 2) / 2.0)
        return torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * env * torch.sin(t)

    def v_exact(self, t, x):
        env = torch.exp(-(x[:, 0:1] ** 2 + x[:, 1:2] ** 2) / 2.0)
        return torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * env * torch.cos(t)

    def g1_terminal(self, x):
        return self.u_exact(torch.full((x.shape[0], 1), self.cfg.T, device=x.device), x)

    def g2_terminal(self, x):
        return self.v_exact(torch.full((x.shape[0], 1), self.cfg.T, device=x.device), x)

    @torch.no_grad()
    def get_analytic_sources(self, t, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        C1, S1 = torch.cos(2 * x1), torch.sin(2 * x1)
        C2, S2 = torch.cos(2 * x2), torch.sin(2 * x2)
        E = torch.exp(-0.5 * (x1 ** 2 + x2 ** 2))
        sin_t, cos_t = torch.sin(t), torch.cos(t)

        u_val = C1 * S2 * E * sin_t
        v_val = S1 * C2 * E * cos_t

        du_dt = C1 * S2 * E * cos_t
        dv_dt = -S1 * C2 * E * sin_t

        u_x1 = (-2 * S1 - x1 * C1) * S2 * E * sin_t
        u_x2 = C1 * (2 * C2 - x2 * S2) * E * sin_t
        v_x1 = (2 * C1 - x1 * S1) * C2 * E * cos_t
        v_x2 = S1 * (-2 * S2 - x2 * C2) * E * cos_t

        lap_u = (((x1 ** 2 - 5) * C1 + 4 * x1 * S1) * S2 + C1 * ((x2 ** 2 - 5) * S2 - 4 * x2 * C2)) * E * sin_t
        lap_v = (((x1 ** 2 - 5) * S1 - 4 * x1 * C1) * C2 + S1 * ((x2 ** 2 - 5) * C2 + 4 * x2 * S2)) * E * cos_t

        mu1, mu2 = x1 - x1 ** 3, x2 - x2 ** 3
        adv_u = mu1 * u_x1 + mu2 * u_x2
        adv_v = mu1 * v_x1 + mu2 * v_x2

        batch = x.shape[0]
        t_exp = t.expand(-1, self.M_total).reshape(-1, 1)

        x_plus_u = (x.unsqueeze(1) + self.y_vec_u.unsqueeze(0)).view(-1, 2)
        x_minus_u = (x.unsqueeze(1) - self.y_vec_u.unsqueeze(0)).view(-1, 2)
        u_plus = self.u_exact(t_exp, x_plus_u).view(batch, -1)
        u_minus = self.u_exact(t_exp, x_minus_u).view(batch, -1)
        I_u = self.K_u * (u_plus + u_minus - 2 * u_val).mean(dim=1, keepdim=True)

        x_plus_v = (x.unsqueeze(1) + self.y_vec_v.unsqueeze(0)).view(-1, 2)
        x_minus_v = (x.unsqueeze(1) - self.y_vec_v.unsqueeze(0)).view(-1, 2)
        v_plus = self.v_exact(t_exp, x_plus_v).view(batch, -1)
        v_minus = self.v_exact(t_exp, x_minus_v).view(batch, -1)
        I_v = self.K_v * (v_plus + v_minus - 2 * v_val).mean(dim=1, keepdim=True)

        f1 = -du_dt - adv_u - 0.5 * lap_u - I_u - torch.sin(v_val)
        f2 = -dv_dt - adv_v - 0.5 * lap_v - I_v - torch.sin(u_val)
        return f1, f2

    def net_fractional_integral(self, net, t, x, var_idx):
        batch = x.shape[0]
        y_vec = self.y_vec_u if var_idx == 0 else self.y_vec_v
        K = self.K_u if var_idx == 0 else self.K_v

        x_plus = (x.unsqueeze(1) + y_vec.unsqueeze(0)).view(-1, 2)
        x_minus = (x.unsqueeze(1) - y_vec.unsqueeze(0)).view(-1, 2)
        t_exp = t.expand(-1, self.M_total).reshape(-1, 1)

        x_combined = torch.cat([x_plus, x_minus], dim=0)
        t_combined = torch.cat([t_exp, t_exp], dim=0)

        pred_u, pred_v = net(t_combined, x_combined)
        pred_all = pred_u if var_idx == 0 else pred_v

        val_plus, val_minus = torch.chunk(pred_all.view(2 * batch, -1), 2, dim=0)
        val_center = net(t, x)[var_idx].view(batch, 1)

        diff = val_plus + val_minus - 2.0 * val_center
        return K * diff.mean(dim=1, keepdim=True)


# ==========================================
# 3. Solver 
# ==========================================
class AdvancedCoupledNet2D(nn.Module):
    def __init__(self, in_dim=3, hidden=128, n_layers=4, sigma_x=0.8, sigma_t=1.0):
        super().__init__()
        self.mapping_size = 256

        self.B_x = nn.Parameter(torch.randn(2, self.mapping_size) * sigma_x, requires_grad=False)
        self.B_t = nn.Parameter(torch.randn(1, self.mapping_size) * sigma_t, requires_grad=False)

        fourier_dim = 4 * self.mapping_size
        self.encoder = nn.Sequential(nn.Linear(fourier_dim, hidden), nn.SiLU())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(n_layers - 1)
        ])

        self.head_u = nn.Linear(hidden, 1)
        self.head_v = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, t, x):
        proj_x = (2.0 * math.pi * x) @ self.B_x
        proj_t = (2.0 * math.pi * t) @ self.B_t
        feat_x = torch.cat([torch.sin(proj_x), torch.cos(proj_x)], dim=-1)
        feat_t = torch.cat([torch.sin(proj_t), torch.cos(proj_t)], dim=-1)

        H = self.encoder(torch.cat([feat_x, feat_t], dim=-1))
        for layer in self.hidden_layers:
            H = H + layer(H)
        return self.head_u(H), self.head_v(H)


class DeepBSDESolver2D:
    def __init__(self, cfg, equation, device):
        self.cfg, self.eq, self.device = cfg, equation, device
        self.net = AdvancedCoupledNet2D(in_dim=3, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs, eta_min=1e-5)

        self.ema_net = copy.deepcopy(self.net)
        self.ema_decay = 0.995

    def update_ema(self):
        with torch.no_grad():
            for p_ema, p_net in zip(self.ema_net.parameters(), self.net.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p_net.data, alpha=1.0 - self.ema_decay)

    def train_epoch(self, current_step):
        batch, device, cfg, eq = self.cfg.batch_size, self.device, self.cfg, self.eq
        dt = cfg.dt

        curriculum_ratio = min(1.0, current_step / (cfg.epochs * 0.5))
        min_start_step = int((cfg.N - 1) * (1.0 - curriculum_ratio))
        n_start = random.randint(min_start_step, cfg.N - 1)

        t = torch.full((batch, 1), n_start * dt, device=device)
        x_u = torch.empty(batch, 2, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        x_v = torch.empty(batch, 2, device=device).uniform_(cfg.x0_low, cfg.x0_high)

        Y_u, _ = self.net(t, x_u)
        _, Y_v = self.net(t, x_v)

        half_batch = batch // 2

        for step in range(n_start, cfg.N):
            x_u_g = x_u.clone().detach().requires_grad_(True)
            x_v_g = x_v.clone().detach().requires_grad_(True)

            u_val_at_xu, v_val_at_xu = self.net(t, x_u_g)
            u_val_at_xv, v_val_at_xv = self.net(t, x_v_g)

            Z_u = torch.autograd.grad(u_val_at_xu.sum(), x_u_g, create_graph=True)[0]
            Z_v = torch.autograd.grad(v_val_at_xv.sum(), x_v_g, create_graph=True)[0]

            dW_u_half = torch.randn(half_batch, 2, device=device) * math.sqrt(dt)
            dW_u = torch.cat([dW_u_half, -dW_u_half], dim=0)

            dW_v_half = torch.randn(half_batch, 2, device=device) * math.sqrt(dt)
            dW_v = torch.cat([dW_v_half, -dW_v_half], dim=0)

            f1_env, _ = eq.get_analytic_sources(t, x_u_g)
            _, f2_env = eq.get_analytic_sources(t, x_v_g)

            net_I_u = eq.net_fractional_integral(self.net, t, x_u_g, var_idx=0)
            net_I_v = eq.net_fractional_integral(self.net, t, x_v_g, var_idx=1)

            drift_Y_u = net_I_u + torch.sin(v_val_at_xu) + f1_env
            drift_Y_v = net_I_v + torch.sin(u_val_at_xv) + f2_env

            Y_u = Y_u - drift_Y_u * dt + torch.sum(Z_u * dW_u, dim=1, keepdim=True) * eq.sigma_u
            Y_v = Y_v - drift_Y_v * dt + torch.sum(Z_v * dW_v, dim=1, keepdim=True) * eq.sigma_v

            x_u = x_u_g.detach() + eq.mu(x_u_g) * dt + eq.sigma_u * dW_u
            x_v = x_v_g.detach() + eq.mu(x_v_g) * dt + eq.sigma_v * dW_v
            t = t + dt

        path_loss = nn.MSELoss()(Y_u, eq.g1_terminal(x_u)) + nn.MSELoss()(Y_v, eq.g2_terminal(x_v))

        x_anchor = torch.empty(batch, 2, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        t_anchor = torch.full((batch, 1), cfg.T, device=device)
        u_anchor, v_anchor = self.net(t_anchor, x_anchor)
        anchor_loss = nn.MSELoss()(u_anchor, eq.g1_terminal(x_anchor)) + nn.MSELoss()(v_anchor,
                                                                                      eq.g2_terminal(x_anchor))

        return path_loss + 10.0 * anchor_loss

    def train(self):
        print(f" 启动 2D Complete System 预热 (Legal Boundary Warm-up)...")
        pre_opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for s in range(1, self.cfg.pretrain_steps + 1):
            pre_opt.zero_grad()
            t_border = torch.full((1024, 1), self.cfg.T, device=self.device)
            x_rand = torch.empty(1024, 2, device=self.device).uniform_(self.cfg.x0_low, self.cfg.x0_high)
            u_pred, v_pred = self.net(t_border, x_rand)
            loss = nn.MSELoss()(u_pred, self.eq.g1_terminal(x_rand)) + nn.MSELoss()(v_pred, self.eq.g2_terminal(x_rand))
            loss.backward()
            pre_opt.step()
            self.update_ema()  
            if s % 500 == 0: print(f'[Boundary Warm-up] step {s:04d} loss={loss.item():.5f}')

        print(f"\n 开始 主阶段训练")
        self.opt.zero_grad()
        for it in range(1, self.cfg.epochs + 1):
            loss_scaled = self.train_epoch(it) / self.cfg.accum_steps
            loss_scaled.backward()

            if it % self.cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
                self.update_ema()  

                real_step = it // self.cfg.accum_steps
                if real_step % 200 == 0:
                    print(
                        f"Update Step {real_step:05d}/{self.cfg.update_steps}  Loss={loss_scaled.item() * self.cfg.accum_steps:.5f}  LR={self.opt.param_groups[0]['lr']:.1e}")


# ==========================================
# 4. 
# ==========================================
if __name__ == "__main__":
    eq = FractionalCoupledComplete2D(cfg)
    solver = DeepBSDESolver2D(cfg, eq, cfg.device)

    t0 = time.time()
    solver.train()
    print(f"\n 训练完毕！总耗时: {time.time() - t0:.1f}s")

    os.makedirs('checkpoints', exist_ok=True)
    weight_path = 'checkpoints/complete_net_2d_fast.pth'
    torch.save(solver.ema_net.state_dict(), weight_path)
    print(f" 权重已成功保存至 {weight_path}")

    print("\n 正在计算相对 L2 误差并生成热力图...")
    solver.ema_net.eval()

    res = 60
    x_lin = np.linspace(cfg.x0_low, cfg.x0_high, res).astype(np.float32)
    X1, X2 = np.meshgrid(x_lin, x_lin)
    x_flat = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=1), device=cfg.device)

    t_final = torch.full((res * res, 1), 1.0, device=cfg.device)

    with torch.no_grad():
        u_pred_final, v_pred_final = solver.ema_net(t_final, x_flat)
        u_exact_f = eq.u_exact(t_final, x_flat)
        v_exact_f = eq.v_exact(t_final, x_flat)

        L2_u_final = torch.sqrt(torch.mean((u_pred_final - u_exact_f) ** 2)) / torch.sqrt(torch.mean(u_exact_f ** 2))
        L2_v_final = torch.sqrt(torch.mean((v_pred_final - v_exact_f) ** 2)) / torch.sqrt(torch.mean(v_exact_f ** 2))

    print("\n" + "=" * 50)
    print(f'相对 L2 误差 - U (t=1.0): {L2_u_final.item():.6e}')
    print(f'相对 L2 误差 - V (t=1.0): {L2_v_final.item():.6e}')
    print("=" * 50 + "\n")

    # --- 3x3 演化热力图生成 ---
    time_steps = [0.0, 0.5, 1.0]


    def evaluate_and_plot_variable(var_name):
        print(f" 正在生成变量 {var_name.upper()} 的 3x3 演化热力图...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        plt.rcParams.update({'font.size': 12})

        for i, t_val in enumerate(time_steps):
            t_eval = torch.full((res * res, 1), t_val, device=cfg.device)

            with torch.no_grad():
                pred_u, pred_v = solver.ema_net(t_eval, x_flat)
                if var_name == 'u':
                    pred = pred_u.cpu().numpy().reshape(res, res)
                    exact = eq.u_exact(t_eval, x_flat).cpu().numpy().reshape(res, res)
                else:
                    pred = pred_v.cpu().numpy().reshape(res, res)
                    exact = eq.v_exact(t_eval, x_flat).cpu().numpy().reshape(res, res)

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
            ax1.set_title(f"Exact {var_name.upper()} (t={t_val})", fontsize=16)
            ax1.set_xlabel("$x_1$")
            ax1.set_ylabel("$x_2$")
            fig.colorbar(im1, ax=ax1)

            ax2 = axes[i, 1]
            im2 = ax2.pcolormesh(X1, X2, pred, cmap='jet', shading='auto')
            ax2.set_title(f"Predicted {var_name.upper()} (t={t_val})", fontsize=16)
            ax2.set_xlabel("$x_1$")
            ax2.set_ylabel("$x_2$")
            fig.colorbar(im2, ax=ax2)

            ax3 = axes[i, 2]
            im3 = ax3.pcolormesh(X1, X2, abs_err, cmap='magma', shading='auto')
            ax3.set_title(err_title, fontsize=16)
            ax3.set_xlabel("$x_1$")
            ax3.set_ylabel("$x_2$")
            fig.colorbar(im3, ax=ax3)

        plt.tight_layout()
        save_name = f'complete_results_{var_name.upper()}_fast.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f" {var_name.upper()} 变量的高清热力图已保存至: {save_name}")


    evaluate_and_plot_variable('u')
    evaluate_and_plot_variable('v')
    plt.show()  