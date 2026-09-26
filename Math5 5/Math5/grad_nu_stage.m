function grad = grad_nu_stage(u_in_cur, u_in_prev, f, clean, params, t, cfg)
%% GRAD_NU_STAGE   FD gradient of stage-t loss w.r.t. nu_t
%
%  nu_t is the gray-level exponent in b(s) = 2*s^nu / (1 + s^nu).
%  Uses symmetric finite differences.
%
%  NOTE: clean is passed directly (not reconstructed from diff).

h = 1e-4;

p_p        = params;
p_p.nu_t(t) = params.nu_t(t) + h;
[u_p, ~]   = forward_stage(u_in_cur, u_in_prev, f, p_p, t, cfg);
L_p        = 0.5 * mean((u_p(:) - clean(:)).^2);

p_m        = params;
p_m.nu_t(t) = max(0.01, params.nu_t(t) - h);
[u_m, ~]   = forward_stage(u_in_cur, u_in_prev, f, p_m, t, cfg);
L_m        = 0.5 * mean((u_m(:) - clean(:)).^2);

grad = (L_p - L_m) / (2 * h);
end
