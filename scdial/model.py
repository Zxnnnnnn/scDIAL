import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import svd


class DBONet_with_E(nn.Module):
    def __init__(
        self,
        mc,
        mf,
        n_view,
        blocks,
        theta_c1,
        theta_c2,
        Zc_init,
        Dc_init,
        Ec_init,
        Zf_init,
        Df_init,
        Ef_init,
        device,
        alpha_temperature=1.0,
        enable_d_perturb=0,
        d_perturb_scale=0.05,
        enable_d_f_perturb=1,
        d_f_perturb_scale=0.05,
        theta_f1=0.01,
        theta_f2=0.001,
        enable_cf_injection=0,
        beta_cf_init=0.01,
        beta_cf_max=0.2,
        cf_gate_init=None,
        feature_to_cell_bridge="x",
    ):
        super(DBONet_with_E, self).__init__()
        self.blocks = blocks
        self.device = device
        self.n_view = n_view
        self.mc = int(mc)
        self.mf = int(mf)
        self.m = self.mc
        self.alpha_temperature = alpha_temperature
        self.enable_d_perturb = bool(enable_d_perturb)
        self.d_perturb_scale = d_perturb_scale
        self.enable_d_f_perturb = bool(enable_d_f_perturb)
        self.d_f_perturb_scale = d_f_perturb_scale
        self.enable_cf_injection = bool(enable_cf_injection)
        self.beta_cf_max = float(beta_cf_max)
        self.cf_gate_max = 0.6
        self.feature_to_cell_bridge = str(feature_to_cell_bridge).lower()
        if self.feature_to_cell_bridge not in {"direct", "x"}:
            raise ValueError(
                "feature_to_cell_bridge must be one of {'direct', 'x'}, "
                f"got {feature_to_cell_bridge}."
            )
        self.register_buffer("Zc_init", torch.tensor(Zc_init, dtype=torch.float32, device=device))
        self.Dc_init = [torch.tensor(d, dtype=torch.float32, device=device) for d in Dc_init]
        self.Ec_init = [torch.tensor(e, dtype=torch.float32, device=device) for e in Ec_init]
        self.theta_c1 = nn.Parameter(torch.FloatTensor([theta_c1]).to(device), requires_grad=True)
        self.theta_c2 = nn.Parameter(torch.FloatTensor([theta_c2]).to(device), requires_grad=True)
        init_alpha = torch.full((n_view,), float(torch.log(torch.tensor(1.0 / n_view))), dtype=torch.float32)
        self.alpha_c_raw = nn.Parameter(init_alpha.to(device), requires_grad=True)
        self.Uc = nn.Linear(self.mc, self.mc).to(device)
        self.Sc = nn.Linear(self.mc, self.mc).to(device)
        self.mu_c = nn.Linear(self.mc, self.mc).to(device)
        self.zita_c = nn.Linear(self.mc, self.mc).to(device)
        self.prelu_c_weight = nn.Parameter(torch.Tensor(self.mc).fill_(0.25).to(device))
        self.dc_mu = nn.ModuleList([nn.Linear(self.mc, self.mc).to(device) for _ in range(n_view)])
        self.dc_sigma = nn.ModuleList([nn.Linear(self.mc, self.mc).to(device) for _ in range(n_view)])

        self.register_buffer("Zf_init", torch.tensor(Zf_init, dtype=torch.float32, device=device))
        self.Df_init = [torch.tensor(d, dtype=torch.float32, device=device) for d in Df_init]
        self.Ef_init = [torch.tensor(e, dtype=torch.float32, device=device) for e in Ef_init]
        self.theta_f1 = nn.Parameter(torch.FloatTensor([theta_f1]).to(device), requires_grad=True)
        self.theta_f2 = nn.Parameter(torch.FloatTensor([theta_f2]).to(device), requires_grad=True)
        self.alpha_f_raw = nn.Parameter(init_alpha.clone().to(device), requires_grad=True)
        self.Uf = nn.Linear(self.mf, self.mf).to(device)
        self.Sf = nn.Linear(self.mf, self.mf).to(device)
        self.mu_f = nn.Linear(self.mf, self.mf).to(device)
        self.zita_f = nn.Linear(self.mf, self.mf).to(device)
        self.prelu_f_weight = nn.Parameter(torch.Tensor(self.mf).fill_(0.25).to(device))
        self.df_mu = nn.ModuleList([nn.Linear(self.mf, self.mf).to(device) for _ in range(n_view)])
        self.df_sigma = nn.ModuleList([nn.Linear(self.mf, self.mf).to(device) for _ in range(n_view)])

        self.n_cell = int(self.Zc_init.shape[1])
        self.n_feature = int(self.Zf_init.shape[1])
        preserve_init_rng = self.feature_to_cell_bridge == "direct"
        if preserve_init_rng:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all()
                if torch.device(device).type == "cuda" and torch.cuda.is_available()
                else None
            )

        self.W_fc = nn.Linear(self.mf, self.mc, bias=True).to(device)
        nn.init.orthogonal_(self.W_fc.weight)
        if self.feature_to_cell_bridge == "direct":
            self.feature_to_cell_rank = self.mf
            self.gene_to_rank = nn.Linear(self.n_feature, self.feature_to_cell_rank, bias=True).to(device)
            self.rank_to_cell = nn.Linear(self.feature_to_cell_rank, self.n_cell, bias=True).to(device)

        if preserve_init_rng:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        gate_init = beta_cf_init if cf_gate_init is None else cf_gate_init
        gate_init = max(min(float(gate_init), self.cf_gate_max - 1e-6), 1e-6)
        gate_ratio = gate_init / self.cf_gate_max
        self.cf_gate = nn.Linear(3 * self.mc, 1).to(device)
        nn.init.zeros_(self.cf_gate.weight)
        with torch.no_grad():
            gate_bias = torch.logit(torch.tensor(gate_ratio, dtype=torch.float32, device=device))
            self.cf_gate.bias.copy_(gate_bias.reshape_as(self.cf_gate.bias))

    def _soft_threshold_c(self, x):
        return F.selu(x - self.theta_c1) - F.selu(-1.0 * x - self.theta_c1)

    def _soft_threshold_f(self, x):
        return F.selu(x - self.theta_f1) - F.selu(-1.0 * x - self.theta_f1)

    def _shrink_global(self, x, theta):
        norm_x = torch.norm(x)
        if norm_x > theta:
            return (norm_x - 1.0 / theta) * x / norm_x
        return torch.zeros_like(x)

    def _refine_dictionary_c(self, d_svd, view_idx):
        if not self.enable_d_perturb:
            return d_svd

        noise = torch.randn_like(d_svd)
        mu = F.prelu(self.dc_mu[view_idx](d_svd), self.prelu_c_weight)
        sigma = F.relu(self.dc_sigma[view_idx](d_svd))
        delta = torch.tanh(mu + sigma * noise) * self.d_perturb_scale
        return d_svd + delta

    def _refine_dictionary_f(self, d_svd, view_idx):
        if not self.enable_d_f_perturb:
            return d_svd

        noise = torch.randn_like(d_svd)
        mu = F.prelu(self.df_mu[view_idx](d_svd), self.prelu_f_weight)
        sigma = F.relu(self.df_sigma[view_idx](d_svd))
        delta = torch.tanh(mu + sigma * noise) * self.d_f_perturb_scale
        return d_svd + delta

    def _svd_uv_numpy(self, c_mat, fallback=None):
        c_mat = torch.nan_to_num(c_mat, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            c_numpy = c_mat.detach().cpu().numpy()
            u_mat, _, vt_mat = svd(c_numpy, full_matrices=False)
            u_mat = torch.tensor(u_mat, dtype=torch.float32, device=c_mat.device)
            vt_mat = torch.tensor(vt_mat, dtype=torch.float32, device=c_mat.device)
            return torch.mm(u_mat, vt_mat)
        except Exception:
            if fallback is not None:
                return fallback
            raise

    def _feature_to_cell(self, zf, features_f, alpha_f):
        if self.feature_to_cell_bridge == "direct":
            h_rank = self.gene_to_rank(zf)
            h_feat = self.rank_to_cell(h_rank)
        else:
            h_feat = torch.zeros(
                (self.mf, features_f[0].shape[0]),
                dtype=zf.dtype,
                device=zf.device,
            )
            for view_idx in range(self.n_view):
                x_bar = features_f[view_idx].t()
                h_feat = h_feat + alpha_f[view_idx] * torch.mm(zf, x_bar)
        zc_feature = self.W_fc(h_feat.t()).t()
        return h_feat, zc_feature

    def _inject_feature_to_cell(self, zc_cell, zc_feature):
        gate_cf = torch.zeros((1, zc_cell.shape[1]), dtype=zc_cell.dtype, device=zc_cell.device)
        if not self.enable_cf_injection:
            return zc_cell, gate_cf, gate_cf.mean(), gate_cf.min(), gate_cf.max()

        zc_feature_norm = F.normalize(zc_feature, p=2, dim=0)
        zc_scale = torch.norm(zc_cell.detach(), p=2, dim=0, keepdim=True)
        zc_feature_scaled = zc_feature_norm * zc_scale
        gate_input = torch.cat(
            [
                zc_cell,
                zc_feature_scaled,
                torch.abs(zc_cell - zc_feature_scaled),
            ],
            dim=0,
        ).t()
        gate_cf = self.cf_gate_max * torch.sigmoid(self.cf_gate(gate_input.detach())).t()
        zc_next = zc_cell + gate_cf * zc_feature_scaled
        return zc_next, gate_cf, gate_cf.mean(), gate_cf.min(), gate_cf.max()

    def _update_zc(self, zc_prev, dc_hist, ec_hist, features_c, alpha_c):
        zc_next = torch.zeros_like(self.Zc_init)
        for view_idx in range(self.n_view):
            input_z = self.Uc(zc_prev.t()).t()
            input_x = torch.mm(dc_hist[view_idx][-1].t(), features_c[view_idx] - ec_hist[view_idx][-1])
            input_x = self.Sc(input_x.t()).t()
            zc_next = zc_next + alpha_c[view_idx] * self._soft_threshold_c(input_z + input_x)
        return zc_next

    def _update_zf(self, zf_prev, df_hist, ef_hist, features_f, alpha_f):
        zf_next = torch.zeros_like(self.Zf_init)
        for view_idx in range(self.n_view):
            input_z = self.Uf(zf_prev.t()).t()
            input_x = torch.mm(df_hist[view_idx][-1].t(), features_f[view_idx] - ef_hist[view_idx][-1])
            input_x = self.Sf(input_x.t()).t()
            zf_next = zf_next + alpha_f[view_idx] * self._soft_threshold_f(input_z + input_x)
        return zf_next

    def forward(self, features_c, features_f, run_feature_branch=True):
        alpha_c = F.softmax(self.alpha_c_raw / self.alpha_temperature, dim=0)
        alpha_f = F.softmax(self.alpha_f_raw / self.alpha_temperature, dim=0)

        zc_hist = [self.Zc_init]
        dc_hist = [[self.Dc_init[i]] for i in range(self.n_view)]
        ec_hist = [[self.Ec_init[i]] for i in range(self.n_view)]
        dc_svd_hist = [[self.Dc_init[i]] for i in range(self.n_view)]

        zf_hist = [self.Zf_init]
        df_hist = [[self.Df_init[i]] for i in range(self.n_view)]
        ef_hist = [[self.Ef_init[i]] for i in range(self.n_view)]
        df_svd_hist = [[self.Df_init[i]] for i in range(self.n_view)]

        h_feat = None
        zc_feature = None
        gate_cf = torch.zeros((1, self.Zc_init.shape[1]), dtype=self.Zc_init.dtype, device=self.Zc_init.device)
        gate_cf_min = gate_cf.min()
        gate_cf_max = gate_cf.max()
        gate_cf_mean = gate_cf.mean()

        for _ in range(self.blocks):
            if run_feature_branch:
                zf_next = self._update_zf(zf_hist[-1], df_hist, ef_hist, features_f, alpha_f)
                zf_hist.append(zf_next)
                h_feat, zc_feature = self._feature_to_cell(zf_hist[-1], features_f, alpha_f)

            zc_cell = self._update_zc(zc_hist[-1], dc_hist, ec_hist, features_c, alpha_c)
            if run_feature_branch:
                zc_next, gate_cf, gate_cf_mean, gate_cf_min, gate_cf_max = self._inject_feature_to_cell(zc_cell, zc_feature)
            else:
                zc_next = zc_cell
            zc_hist.append(zc_next)

            for view_idx in range(self.n_view):
                c_mat = torch.mm(features_c[view_idx], zc_hist[-1].t()) - torch.mm(
                    ec_hist[view_idx][-1], zc_hist[-1].t()
                )
                dc_svd = self._svd_uv_numpy(c_mat, fallback=dc_hist[view_idx][-1])
                dc_svd_hist[view_idx].append(dc_svd)
                dc_hist[view_idx].append(self._refine_dictionary_c(dc_svd, view_idx))

            for view_idx in range(self.n_view):
                residual = features_c[view_idx] - torch.mm(dc_hist[view_idx][-1], zc_hist[-1])
                ec_hist[view_idx].append(self._shrink_global(residual, self.theta_c2))

            if run_feature_branch:
                for view_idx in range(self.n_view):
                    c_mat = torch.mm(features_f[view_idx], zf_hist[-1].t()) - torch.mm(
                        ef_hist[view_idx][-1], zf_hist[-1].t()
                    )
                    df_svd = self._svd_uv_numpy(c_mat, fallback=df_hist[view_idx][-1])
                    df_svd_hist[view_idx].append(df_svd)
                    df_hist[view_idx].append(self._refine_dictionary_f(df_svd, view_idx))

                for view_idx in range(self.n_view):
                    residual = features_f[view_idx] - torch.mm(df_hist[view_idx][-1], zf_hist[-1])
                    ef_hist[view_idx].append(self._shrink_global(residual, self.theta_f2))

        if run_feature_branch:
            if h_feat is None or zc_feature is None:
                h_feat, zc_feature = self._feature_to_cell(zf_hist[-1], features_f, alpha_f)
            zf_out = zf_hist[-1]
            df_out = [df_hist[i][-1] for i in range(self.n_view)]
            ef_out = [ef_hist[i][-1] for i in range(self.n_view)]
            df_svd_out = [df_svd_hist[i][-1] for i in range(self.n_view)]
        else:
            zf_out = self.Zf_init
            df_out = [self.Df_init[i] for i in range(self.n_view)]
            ef_out = [self.Ef_init[i] for i in range(self.n_view)]
            df_svd_out = [self.Df_init[i] for i in range(self.n_view)]
            h_feat = torch.zeros((self.mf, self.Zc_init.shape[1]), dtype=self.Zc_init.dtype, device=self.Zc_init.device)
            zc_feature = torch.zeros_like(self.Zc_init)

        return {
            "Zc": zc_hist[-1],
            "Dc": [dc_hist[i][-1] for i in range(self.n_view)],
            "Ec": [ec_hist[i][-1] for i in range(self.n_view)],
            "alpha_c": alpha_c,
            "Dc_svd": [dc_svd_hist[i][-1] for i in range(self.n_view)],
            "Zf": zf_out,
            "Df": df_out,
            "Ef": ef_out,
            "alpha_f": alpha_f,
            "Df_svd": df_svd_out,
            "Hf": h_feat,
            "Zc_hat": zc_feature,
            "gate_cf": gate_cf,
            "gate_cf_mean": gate_cf_mean,
            "gate_cf_min": gate_cf_min,
            "gate_cf_max": gate_cf_max,
            "beta_cf": gate_cf_mean,
        }
