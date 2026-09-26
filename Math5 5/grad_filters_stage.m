function grad_K = grad_filters_stage(u_in_cur, u_in_prev, f, clean, params, t, cfg)
%% GRAD_FILTERS_STAGE  Analytic filter gradient for a single 2-D patch.
%
%  Call once per patch and average over mini-batch in train_stagewise.
%  Inputs must ALL be 2-D arrays [H x W].
%
%  Two-term chain rule (TNRD supplemental §2.1, Eq.10):
%
%  D_i = (1/√Nk) rot180(k_i) ★ w_i,   w_i = g_w ⊙ φ_i(z_i),   z_i = k_i ★ u
%
%  Term 1 — gradient through OUTPUT conv rot180(k_i) ★ w_i:
%    ∂L/∂rot180(k_i) = xcorr_fs(w_i, δ_D) / √Nk
%    ∂L/∂k_i        += rot180(above)
%
%  Term 2 — gradient through INPUT conv k_i ★ u:
%    δ_w_i = (rot180(k_i))^T ★ δ_D / √Nk  = k_i ★ δ_D / √Nk
%    δ_z_i = g_w ⊙ φ'_i(z_i) ⊙ δ_w_i
%    ∂L/∂k_i += xcorr_fs(u, δ_z_i)
%
%  xcorr_fs produces [fs × fs] correctly via:
%    padarray(A, [half half], 'symmetric') then conv2(..., rot90(B,2), 'valid')
%    Output size = (H+fs-1 - H + 1) = fs  ✓
%
%  CRITICAL BUG FIXED: conv2(HxW, HxW, 'valid') = [1x1] scalar (just a dot
%  product), NOT the [fs x fs] filter gradient.  The padding step is essential.

%% Validate 2-D inputs
if ndims(u_in_cur) > 2 || ndims(clean) > 2                         %#ok<ISMAT>
    error('grad_filters_stage: inputs must be 2-D. Got %d-D.', ndims(u_in_cur));
end

fs    = cfg.fsize;
half  = (fs - 1) / 2;
Nk    = params.Nk;
tau   = cfg.tau;
gamma = cfg.gamma_fix;
eps_f = cfg.epsilon;
eps_fid = 0.01;   % must match forward_stage

nu = max(0.3,  params.nu_t(t));
Ke = max(0.05, params.Kedge_t(t));

coeff_next = 1 + gamma * tau;

%% Recompute g_w (must match forward_stage exactly, using 2-D gradient)
u_xi = imgaussfilt(double(u_in_cur), 1.0);
M_xi = max(u_xi(:)) + eps_f;
s    = max(eps_f, min(1-eps_f, u_xi / M_xi));
b_u  = 2*(s.^nu) ./ (1 + s.^nu);
[gy, gx] = gradient(u_xi);
c_u  = 1 ./ (1 + (sqrt(gx.^2 + gy.^2 + eps_f) ./ Ke).^2);
g_w  = b_u .* c_u;

%% Forward pass → error signal δ
[u_next, ~] = forward_stage(u_in_cur, u_in_prev, f, params, t, cfg);
N_pix  = numel(u_next);
dL_du  = (u_next - clean) / N_pix;

%% Backprop through telegraph update
%   u_next = (... − τ²·R(u) ...) / (1+γτ)
%   ∂L/∂R  = (−τ²/coeff_next) · dL/du_next
delta_D = (-tau^2 / coeff_next) * dL_du;   % [H × W]

%% Pre-pad u for xcorr → ensures output is exactly [fs × fs]
u_pad = padarray(double(u_in_cur), [half half], 'symmetric');

K_filt   = params.K_t{t};
phi_type = params.phi_type;
grad_K   = zeros(fs, fs, Nk);

for i = 1:Nk
    ki   = K_filt(:,:,i);
    ki_r = rot90(ki, 2);   % rot180(k_i)

    %% Filter response and influence function
    z_i    = imfilter(double(u_in_cur), ki,  'symmetric', 'conv');
    phi_z  = apply_phi(z_i,  phi_type);    % φ_i(z_i)
    dphi_z = apply_dphi(z_i, phi_type);    % φ'_i(z_i)
    w_i    = g_w .* phi_z;                 % [H × W]

    %% Term 1 — gradient through OUTPUT conv rot180(k_i) ★ w_i
    %   ∂L/∂rot180(k_i) = xcorr_fs(w_i, δ_D) / √Nk
    %   ∂L/∂k_i  = rot180(above)
    w_pad = padarray(double(w_i), [half half], 'symmetric');
    xc1   = conv2(w_pad, rot90(double(delta_D), 2), 'valid') / sqrt(Nk);
    grad1 = rot90(xc1, 2);   % [fs × fs]

    %% Term 2 — gradient through INPUT conv k_i ★ u
    %   δ_w_i = adjoint of (rot180(k_i) ★ ·) applied to δ_D
    %         = k_i ★ δ_D / √Nk   [because adjoint of rot180(k_i) conv is k_i conv]
    delta_wi = imfilter(double(delta_D), ki, 'symmetric', 'conv') / sqrt(Nk);
    delta_zi = g_w .* dphi_z .* delta_wi;
    xc2      = conv2(u_pad, rot90(double(delta_zi), 2), 'valid');
    grad2    = xc2;   % [fs × fs]

    grad_K(:,:,i) = grad1 + grad2;
end
end