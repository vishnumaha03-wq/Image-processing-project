function [u_next, u_cur_out] = forward_stage(u_cur, u_prev, f, params, t, cfg)
%% FORWARD_STAGE  One unrolled Telegraph-TNRD stage  (2-D images ONLY)
%
%  Telegraph update (Majee 2020, Eq.2.5 discretised):
%
%   (1+γτ)u^{n+1} = (2+γτ)u^n  −  u^{n-1}
%                 −  τ² R(u^n)            [TNRD regularisation]
%                 −  τ² λ F(u^n, f)       [multiplicative fidelity]
%
%  Regularisation (TNRD Chen & Pock 2017, 1/sqrt(Nk) normalisation):
%    R(u) = (1/√Nk) Σ_i  rot180(k_i) ★ (g_w ⊙ φ_i(k_i ★ u))
%
%  Diffusion weight  g_w = b(u_ξ) · c(|∇u_ξ|):
%    b(s) = 2s^ν/(1+s^ν),  s = u_ξ/M_ξ         (gray-level indicator)
%    c(r) = 1/(1+(r/Ke)²)                         (edge detector)
%
%  Multiplicative Gamma fidelity gradient:
%    F(u,f) = (u − f) / (u² + ε_fid)
%    Derived from MAP energy L·u + f/u (gamma likelihood, L absorbed into λ)
%    ε_fid = 0.01  (NOT cfg.epsilon) — prevents explosion in dark regions
%
%  CRITICAL: This function is 2-D ONLY.  Passing 4-D batch arrays causes
%  gradient() to compute wrong spatial derivatives, breaking g_w and all
%  downstream gradients.  Training loops over patches individually.

%% Validate — catch 4-D misuse immediately
if ndims(u_cur) > 2                                                 %#ok<ISMAT>
    error(['forward_stage: received %d-D array. ' ...
           'Must be 2-D. Loop over batch samples in train_stagewise.'], ...
          ndims(u_cur));
end

gamma = cfg.gamma_fix;
tau   = cfg.tau;
eps_f = cfg.epsilon;

%% Parameter extraction with enforced bounds
lam = max(0.01,  params.lambda_t(t));   % fidelity weight
nu  = max(0.3,   params.nu_t(t));       % gray-level exponent
Ke  = max(0.05,  params.Kedge_t(t));    % edge threshold
Nk  = params.Nk;

coeff_next = 1 + gamma * tau;
coeff_cur  = 2 + gamma * tau;

%% Step 1 — Gaussian smoothing  u_ξ = G_σ ★ u  (σ=1 fixed, paper §2)
u_xi = imgaussfilt(double(u_cur), 1.0);   % imgaussfilt: safe 2-D only

%% Step 2 — Gray-level indicator  b(s) = 2s^ν/(1+s^ν)
%  s = u_ξ/M_ξ ∈ (0,1]: dark pixels → b→0 (less diffusion)
M_xi = max(u_xi(:)) + eps_f;
s    = max(eps_f, min(1-eps_f, u_xi / M_xi));
b_u  = 2*(s.^nu) ./ (1 + s.^nu);

%% Step 3 — Edge detector  c(|∇u_ξ|/Ke)
%  gradient() called on 2-D array: returns [Gy, Gx] correctly
[gy, gx] = gradient(u_xi);
gmag      = sqrt(gx.^2 + gy.^2 + eps_f);
c_u       = 1 ./ (1 + (gmag ./ Ke).^2);
g_w       = b_u .* c_u;      % ∈ (0,1), small at dark edges

%% Step 4 — TNRD regularisation
%  R(u) = (1/√Nk) Σ_i rot180(k_i) ★ (g_w ⊙ φ_i(k_i ★ u))
%  1/√Nk normalisation: Chen & Pock 2017, Sec.3.1
K_filt   = params.K_t{t};
phi_type = params.phi_type;
reg      = zeros(size(u_cur));
for i = 1:Nk
    ki    = K_filt(:,:,i);
    r_i   = imfilter(double(u_cur), ki, 'symmetric', 'conv');
    phi_r = apply_phi(r_i, phi_type);
    reg   = reg + imfilter(g_w .* phi_r, rot90(ki,2), 'symmetric', 'conv');
end
reg = reg / sqrt(Nk);

%% Step 5 — Multiplicative Gamma fidelity gradient
%  F(u,f) = (u − f) / (u² + ε_fid)
%
%  WHY ε_fid = 0.01 (not cfg.epsilon=1e-6):
%    With ε=1e-6 and u=0.05:  F = (0.05-f)/(0.0025+1e-6) ≈ 400*(0.05-f)
%    With tau=0.1, tau²·λ·F ≈ 0.01·1·400·|0.05-f| → EXPLOSIVE
%    With ε=0.01:            F = (0.05-f)/(0.0025+0.01) ≈ 50*(0.05-f)
%    Contribution ≤ 0.01·1·50·1 = 0.5 → bounded
eps_fid = 0.01;
u_safe  = max(u_cur, sqrt(eps_fid));    % minimum u before squaring
fid     = (u_safe - f) ./ (u_safe.^2 + eps_fid);
fid     = max(-50, min(50, fid));       % hard clip: prevents any single pixel exploding

%% Step 6 — Telegraph update
%  Sign: −τ²λF because F = ∇E_data, and we descend the energy gradient
u_next = (coeff_cur .* u_cur   ...
          - u_prev              ...
          - tau^2 .* reg        ...
          - tau^2 .* lam .* fid) / coeff_next;

u_next    = max(eps_f, min(1.0, u_next));
u_cur_out = u_cur;
end