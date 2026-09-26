function grad_K = grad_filters_stage(u_in_cur, u_in_prev, f, clean, params, t, cfg)
%% GRAD_FILTERS_STAGE  Analytic gradient w.r.t. filter bank K_t{t}
%
%  TNRD regularization:
%    reg = (1/Nk) * sum_i  K_i^T * (g .* phi(K_i * u))
%
%  Analytic gradient via chain rule:
%    dL/dK_i = (1/Nk) * [
%       conv2(u, flip(g .* phi'(r_i) .* delta_reg), 'valid')  [term1]
%     + conv2(delta_reg_back, flip(g .* phi(r_i)), 'valid')    [term2]
%    ]
%  where delta_reg = dL/du_next * d(u_next)/d(reg)

eps_f    = cfg.epsilon;
tau      = cfg.tau;
gamma    = cfg.gamma_fix;
Nk       = params.Nk;
phi_type = params.phi_type;
K_filt   = params.K_t{t};
nu       = params.nu_t(t);
Ke       = max(params.Kedge_t(t), 0.01);
fs       = cfg.fsize;

coeff_next = 1 + gamma * tau;

%% Compute g_w (same as forward pass)
h_g  = fspecial('gaussian', [5 5], 1.0);
u_xi = convn(u_in_cur, h_g, 'same');
M_xi = max(u_xi(:)) + eps_f;
s    = max(eps_f, min(1-eps_f, u_xi/M_xi));
b_u  = 2*(s.^nu)./(1+s.^nu);
[gy,gx] = gradient(u_xi);
gmag = sqrt(gx.^2+gy.^2+eps_f);
c_u  = 1./(1+(gmag/Ke).^2);
g_w  = b_u .* c_u;

%% Forward pass
[u_next,~] = forward_stage(u_in_cur, u_in_prev, f, params, t, cfg);

%% Loss gradient w.r.t. u_next
dL_du = (u_next - clean) / numel(u_next);

%% Backprop through telegraph update
%  u_next = (... - tau^2 * reg ...) / coeff_next
%  => dL/d(reg) = dL/du_next * (-tau^2/coeff_next)
delta = dL_du * (-tau^2 / coeff_next);

%% Analytic gradient for each filter
grad_K = zeros(fs, fs, Nk);

for i = 1:Nk
    ki    = K_filt(:,:,i);
    r_i   = convn(u_in_cur, ki, 'same');
    phi_r = apply_phi(r_i,  phi_type);
    dph_r = apply_dphi(r_i, phi_type);

    % Weighted residuals
    w1 = g_w .* dph_r .* delta / Nk;  % for term1
    w2 = g_w .* phi_r / Nk;            % for term2

    % Term 1: gradient through inner conv K_i * u
    % dL/dK_i += corr(u_in_cur, w1)
    % Use conv with flipped kernel = correlation
    t1 = conv2(squeeze(mean(u_in_cur,4)), squeeze(mean(w1,4)), 'valid');

    % Term 2: gradient through outer conv K_i^T * (...)
    % dL/dK_i += corr(delta_backprop, w2)
    % delta backprop through K_i^T: convolve delta with w2
    t2 = conv2(squeeze(mean(w2,4)), squeeze(mean(delta,4)), 'valid');

    % Combine and resize to [fs x fs]
    g_combined = t1 + t2;

    % Handle size mismatch
    [r_g, c_g] = size(g_combined);
    if r_g == fs && c_g == fs
        grad_K(:,:,i) = g_combined;
    elseif r_g >= fs && c_g >= fs
        % Trim to filter size
        r_off = floor((r_g - fs)/2);
        c_off = floor((c_g - fs)/2);
        grad_K(:,:,i) = g_combined(r_off+1:r_off+fs, c_off+1:c_off+fs);
    else
        % Pad to filter size
        grad_K(:,:,i) = padarray(g_combined, ...
            [ceil((fs-r_g)/2) ceil((fs-c_g)/2)], 0, 'both');
        grad_K(:,:,i) = grad_K(1:fs, 1:fs, i);
    end
end
end