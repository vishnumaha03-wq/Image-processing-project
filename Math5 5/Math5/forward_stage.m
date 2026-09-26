function [u_next, u_cur_out] = forward_stage(u_cur, u_prev, f, params, t, cfg)
%% FORWARD_STAGE  One unrolled Telegraph-TNRD stage
%
%  Telegraph PDE (Majee et al. 2020):
%    (1+gamma*tau)*u^{n+1} = (2+gamma*tau)*u^n - u^{n-1}
%                          - tau^2 * reg(u^n)
%                          - tau^2 * lambda * (u - f)/u^2
%
%  Fidelity derivation for multiplicative gamma noise (Aubert & Aujol 2008):
%    MAP energy:  E(u) = log(u) + f/u
%    dE/du      = 1/u - f/u^2  =  (u - f) / u^2
%    So the update subtracts lambda * (u-f)/u^2
%
%  Accepts 2-D [H,W] or 4-D [H,W,1,N] batch input.

%% ── 4-D batch dispatch ───────────────────────────────────────────────────
if ndims(u_cur) == 4                                    %#ok<ISMAT>
    N         = size(u_cur, 4);
    u_next    = zeros(size(u_cur), 'double');
    u_cur_out = zeros(size(u_cur), 'double');
    for n = 1:N
        [u_next(:,:,1,n), u_cur_out(:,:,1,n)] = forward_stage( ...
            u_cur(:,:,1,n), u_prev(:,:,1,n), f(:,:,1,n), params, t, cfg);
    end
    return
end

%% ── Squeeze to 2-D double ───────────────────────────────────────────────
u_cur  = double(squeeze(u_cur));
u_prev = double(squeeze(u_prev));
f      = double(squeeze(f));

%% ── Unpack parameters ───────────────────────────────────────────────────
gamma = cfg.gamma_fix;
tau   = cfg.tau;
eps_f = cfg.epsilon;
lam   = params.lambda_t(t);
nu    = params.nu_t(t);
Ke    = max(params.Kedge_t(t), 0.01);
Nk    = params.Nk;

coeff_next = 1 + gamma * tau;
coeff_cur  = 2 + gamma * tau;

%% ── Step 1: Gaussian smoothing  u_xi = G_sigma * u ─────────────────────
h_g  = fspecial('gaussian', [5 5], 1.0);
u_xi = imfilter(u_cur, h_g, 'replicate');

%% ── Step 2: Gray-level indicator  b(s) = 2*s^nu / (1 + s^nu) ───────────
%  s = u_xi / M_xi  in [0, 1]
M_xi = max(u_xi(:)) + eps_f;
s    = max(eps_f, min(1 - eps_f, u_xi / M_xi));
b_u  = 2 .* (s .^ nu) ./ (1 + s .^ nu);

%% ── Step 3: Edge detector  c(r) = 1 / (1 + (|grad u_xi| / Ke)^2) ───────
[gy, gx] = gradient(u_xi);
gmag     = sqrt(gx .^ 2 + gy .^ 2 + eps_f);
c_u      = 1 ./ (1 + (gmag / Ke) .^ 2);

%% ── Step 4: Combined diffusion coefficient  g = b(u) * c(|grad u|) ──────
g_w = b_u .* c_u;

%% ── Step 5: TNRD regularisation ─────────────────────────────────────────
%  reg = (1/sqrt(Nk)) * sum_i  k_bar_i * (g_w .* phi_i(k_i * u))
%  1/sqrt(Nk) normalisation matches TNRD paper (Chen & Pock 2017, Sec 3.1)
K_filt   = params.K_t{t};
phi_type = params.phi_type;
reg      = zeros(size(u_cur));

for i = 1:Nk
    ki    = K_filt(:,:,i);
    r_i   = conv2(u_cur, ki, 'same');
    phi_r = apply_phi(r_i, phi_type);
    reg   = reg + conv2(g_w .* phi_r, rot90(ki, 2), 'same');
end
reg = reg / sqrt(Nk);

%% ── Step 6: Fidelity gradient for multiplicative gamma noise ────────────
%  Correct form: dE/du = (u - f) / u^2
%  NOT (u - f)  — that is Gaussian/additive
%  NOT (1 - f/u) — that is the RLO constraint, not the MAP gradient
u_safe   = max(u_cur, eps_f);
fidelity = (u_safe - f) ./ (u_safe .^ 2 + eps_f);

%% ── Step 7: Telegraph update ────────────────────────────────────────────
u_next = (coeff_cur .* u_cur   ...
          - u_prev              ...
          - tau^2  .* reg       ...
          - tau^2  .* lam .* fidelity) / coeff_next;

u_next    = max(eps_f, min(1.0, u_next));
u_cur_out = u_cur;
end