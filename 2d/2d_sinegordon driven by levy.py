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


class FractionalCoupledSineGordon2D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.C1, self.C2 = float(C_2D_alpha(cfg.alpha1)), float(C_2D_alpha(cfg.alpha2))
        r = cfg.levy_eps
        var_1 = self.C1 * math.pi * (r ** (2 - cfg.alpha1)) / (2 - cfg.alpha1)
        var_2 = self.C2 * math.pi * (r ** (2 - cfg.alpha2)) / (2 - cfg.alpha2)
        self.sigma_u, self.sigma_v = math.sqrt(var_1), math.sqrt(var_2)

    def g1_terminal(self, x):
        return torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * math.sin(self.cfg.T)

    def g2_terminal(self, x):
        return torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * math.cos(self.cfg.T)

    def u_exact(self, t, x):
        return torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * torch.sin(t)

    def v_exact(self, t, x):
        return torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * torch.cos(t)

    def mc_fractional_integral_2d(self, net, t, x, alpha, C_alpha, var_idx):
        device, batch = x.device, x.shape[0]
        M_rho, M_theta = self.cfg.levy_M_rho, self.cfg.levy_M_theta
        r, R = float(self.cfg.levy_eps), float(self.cfg.levy_R)

        u_rho = torch.linspace(0.5 / M_rho, 1.0 - 0.5 / M_rho, M_rho, device=device)
        rho = (r ** (-alpha) - u_rho * (r ** (-alpha) - R ** (-alpha))).pow(-1.0 / alpha)
        theta = torch.linspace(0.5 / M_theta, 1.0 - 0.5 / M_theta, M_theta, device=device) * math.pi
        rho_grid, theta_grid = torch.meshgrid(rho, theta, indexing='ij')

        y_vec = torch.stack([rho_grid.reshape(-1) * torch.cos(theta_grid.reshape(-1)),
                             rho_grid.reshape(-1) * torch.sin(theta_grid.reshape(-1))], dim=1)

        x_plus_y = (x.unsqueeze(1) + y_vec.unsqueeze(0)).view(-1, 2)
        x_minus_y = (x.unsqueeze(1) - y_vec.unsqueeze(0)).view(-1, 2)
        t_exp = t.expand(-1, M_rho * M_theta).reshape(-1, 1)

        x_combined = torch.cat([x_plus_y, x_minus_y], dim=0)
        t_combined = torch.cat([t_exp, t_exp], dim=0)

        pred_u, pred_v = net(t_combined, x_combined)
        pred_all = pred_u if var_idx == 0 else pred_v

        val_plus, val_minus = torch.chunk(pred_all.view(2 * batch, -1), 2, dim=0)
        val_center = net(t, x)[var_idx].view(batch, 1)

        diff = val_plus + val_minus - 2.0 * val_center
        K = C_alpha * ((r ** (-alpha) - R ** (-alpha)) / alpha) * math.pi
        return K * diff.mean(dim=1, keepdim=True)

    def f1(self, net, t, x, v_at_x):
        integral_u = self.mc_fractional_integral_2d(net, t, x, self.cfg.alpha1, self.C1, var_idx=0)
        ue, ve = self.u_exact(t, x), self.v_exact(t, x)
        ute = torch.cos(2.0 * x[:, 0:1]) * torch.sin(2.0 * x[:, 1:2]) * torch.cos(t)
        eigen_val = 8.0 ** (self.cfg.alpha1 / 2.0)
        return integral_u + torch.sin(v_at_x) + (eigen_val * ue - ute - torch.sin(ve))

    def f2(self, net, t, x, u_at_x):
        integral_v = self.mc_fractional_integral_2d(net, t, x, self.cfg.alpha2, self.C2, var_idx=1)
        ue, ve = self.u_exact(t, x), self.v_exact(t, x)
        vte = -torch.sin(2.0 * x[:, 0:1]) * torch.cos(2.0 * x[:, 1:2]) * torch.sin(t)
        eigen_val = 8.0 ** (self.cfg.alpha2 / 2.0)
        return integral_v + torch.sin(u_at_x) + (eigen_val * ve - vte - torch.sin(ue))


# ==========================================
# 3. Solver 
# ==========================================
class AdvancedCoupledNet2D(nn.Module):
    def __init__(self, in_dim=3, hidden=128, n_layers=4, sigma=0.8):
        super().__init__()
        self.mapping_size = 256
        self.B = nn.Parameter(torch.randn(in_dim, self.mapping_size) * sigma, requires_grad=False)
        fourier_dim = 2 * self.mapping_size

        self.encoder = nn.Sequential(nn.Linear(fourier_dim, hidden), nn.SiLU())
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()))

        self.head_u = nn.Linear(hidden, 1)
        self.head_v = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, t, x):
        proj = (2.0 * math.pi * torch.cat([t, x], dim=1)) @ self.B
        inp_fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        H = self.encoder(inp_fourier)
        for layer in self.hidden_layers:
            H = H + layer(H)
        return self.head_u(H), self.head_v(H)


class DeepBSDESolver2D:
    def __init__(self, cfg, equation, device):
        self.cfg, self.eq, self.device = cfg, equation, device
        self.net = AdvancedCoupledNet2D(in_dim=3, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2500, gamma=0.8)

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

        for step in range(n_start, cfg.N):
            x_u_g = x_u.clone().detach().requires_grad_(True)
            x_v_g = x_v.clone().detach().requires_grad_(True)

            u_val_at_xu, v_val_at_xu = self.net(t, x_u_g)
            u_val_at_xv, v_val_at_xv = self.net(t, x_v_g)

            Z_u = torch.autograd.grad(u_val_at_xu.sum(), x_u_g, create_graph=True)[0]
            Z_v = torch.autograd.grad(v_val_at_xv.sum(), x_v_g, create_graph=True)[0]

            dW_u = torch.randn(batch, 2, device=device) * math.sqrt(dt)
            dW_v = torch.randn(batch, 2, device=device) * math.sqrt(dt)

            Y_u = Y_u - eq.f1(self.net, t, x_u_g, v_val_at_xu) * dt + torch.sum(Z_u * dW_u, dim=1,
                                                                                keepdim=True) * eq.sigma_u
            Y_v = Y_v - eq.f2(self.net, t, x_v_g, u_val_at_xv) * dt + torch.sum(Z_v * dW_v, dim=1,
                                                                                keepdim=True) * eq.sigma_v

            x_u = x_u_g.detach() + eq.sigma_u * dW_u
            x_v = x_v_g.detach() + eq.sigma_v * dW_v
            t = t + dt

        term_u_exact = eq.g1_terminal(x_u)
        term_v_exact = eq.g2_terminal(x_v)
        path_loss = nn.MSELoss()(Y_u, term_u_exact) + nn.MSELoss()(Y_v, term_v_exact)

        x_anchor = torch.empty(batch, 2, device=device).uniform_(cfg.x0_low, cfg.x0_high)
        t_anchor = torch.full((batch, 1), cfg.T, device=device)
        u_anchor, v_anchor = self.net(t_anchor, x_anchor)
        anchor_loss = nn.MSELoss()(u_anchor, eq.g1_terminal(x_anchor)) + nn.MSELoss()(v_anchor,
                                                                                      eq.g2_terminal(x_anchor))

        return path_loss + 10.0 * anchor_loss

    def train(self):
        print(f" 启动 2D Sine-Gordon 预热 (Constant Lifting Warm-up)...")
        pre_opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for s in range(1, self.cfg.pretrain_steps + 1):
            pre_opt.zero_grad()
            t_rand = torch.rand(1024, 1, device=self.device) * self.cfg.T
            x_rand = torch.empty(1024, 2, device=self.device).uniform_(self.cfg.x0_low, self.cfg.x0_high)
            u_pred, v_pred = self.net(t_rand, x_rand)
            loss = nn.MSELoss()(u_pred, self.eq.g1_terminal(x_rand)) + nn.MSELoss()(v_pred, self.eq.g2_terminal(x_rand))
            loss.backward();
            pre_opt.step()
            if s % 500 == 0: print(f'[Legal Warm-up] step {s:04d} loss={loss.item():.5f}')

        print(f"\n 开始 2D Sine-Gordon 主阶段训练 ")
        self.opt.zero_grad()
        for it in range(1, self.cfg.epochs + 1):
            loss_scaled = self.train_epoch(it) / self.cfg.accum_steps
            loss_scaled.backward()

            if it % self.cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step();
                self.opt.zero_grad();
                self.scheduler.step()

                real_step = it // self.cfg.accum_steps
                if real_step % 200 == 0:
                    print(
                        f"Update Step {real_step:05d}/{self.cfg.update_steps}  Loss={loss_scaled.item() * self.cfg.accum_steps:.5f}  LR={self.opt.param_groups[0]['lr']:.1e}")


# ==========================================
# 4. 
# ==========================================
if __name__ == "__main__":
    eq = FractionalCoupledSineGordon2D(cfg)
    solver = DeepBSDESolver2D(cfg, eq, cfg.device)

    t0 = time.time()
    solver.train()
    print(f"\n 训练完毕！总耗时: {time.time() - t0:.1f}s")

    os.makedirs('checkpoints', exist_ok=True)
    weight_path = 'checkpoints/sine_gordon_net_2d.pth'
    torch.save(solver.net.state_dict(), weight_path)
    print(f" 权重已成功保存至 {weight_path}")

    # --- 最终评估模块 ---
    print("\n 正在计算最终相对 L2 误差并生成热力图...")
    solver.net.eval()

    res = 60
    x_lin = np.linspace(cfg.x0_low, cfg.x0_high, res).astype(np.float32)
    X1, X2 = np.meshgrid(x_lin, x_lin)
    x_flat = torch.tensor(np.stack([X1.ravel(), X2.ravel()], axis=1), device=cfg.device)

    t_final = torch.full((res * res, 1), 1.0, device=cfg.device)

    with torch.no_grad():
        u_pred_final, v_pred_final = solver.net(t_final, x_flat)
        u_pred_f = u_pred_final.cpu().numpy().reshape(res, res)
        v_pred_f = v_pred_final.cpu().numpy().reshape(res, res)

        u_exact_f = eq.u_exact(t_final, x_flat).cpu().numpy().reshape(res, res)
        v_exact_f = eq.v_exact(t_final, x_flat).cpu().numpy().reshape(res, res)

    L2_u_final = np.sqrt(np.mean((u_pred_f - u_exact_f) ** 2)) / np.sqrt(np.mean(u_exact_f ** 2))
    L2_v_final = np.sqrt(np.mean((v_pred_f - v_exact_f) ** 2)) / np.sqrt(np.mean(v_exact_f ** 2))

    print("\n" + "=" * 50)
    print(f' 2D Sine-Gordon 相对 L2 误差 - U (t=1.0): {L2_u_final:.6e}')
    print(f' 2D Sine-Gordon 相对 L2 误差 - V (t=1.0): {L2_v_final:.6e}')
    print("=" * 50 + "\n")

    
    time_steps = [0.0, 0.5, 1.0]


    def evaluate_and_plot_variable(var_name):
        print(f" 正在生成变量 {var_name.upper()} 的 3x3 演化热力图...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        plt.rcParams.update({'font.size': 12})

        for i, t_val in enumerate(time_steps):
            t_eval = torch.full((res * res, 1), t_val, device=cfg.device)

            with torch.no_grad():
                pred_u, pred_v = solver.net(t_eval, x_flat)
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

            # 精确解
            ax1 = axes[i, 0]
            im1 = ax1.pcolormesh(X1, X2, exact, cmap='jet', shading='auto')
            ax1.set_title(f"Exact {var_name.upper()} (t={t_val})", fontsize=16)
            ax1.set_xlabel("$x_1$");
            ax1.set_ylabel("$x_2$")
            fig.colorbar(im1, ax=ax1)

            # 预测解
            ax2 = axes[i, 1]
            im2 = ax2.pcolormesh(X1, X2, pred, cmap='jet', shading='auto')
            ax2.set_title(f"Predicted {var_name.upper()} (t={t_val})", fontsize=16)
            ax2.set_xlabel("$x_1$");
            ax2.set_ylabel("$x_2$")
            fig.colorbar(im2, ax=ax2)

            # 绝对误差
            ax3 = axes[i, 2]
            im3 = ax3.pcolormesh(X1, X2, abs_err, cmap='magma', shading='auto')
            ax3.set_title(err_title, fontsize=16)
            ax3.set_xlabel("$x_1$");
            ax3.set_ylabel("$x_2$")
            fig.colorbar(im3, ax=ax3)

        plt.tight_layout()
        save_name = f'sine_gordon_results_{var_name.upper()}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"🎉 {var_name.upper()} 变量的高清热力图已保存至: {save_name}")


    evaluate_and_plot_variable('u')
    evaluate_and_plot_variable('v')
    plt.show()