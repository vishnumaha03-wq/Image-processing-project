function grad = grad_Kedge_stage(u_in_cur, u_in_prev, f, clean, params, t, cfg)
%% GRAD_KEDGE_STAGE   FD gradient of stage-t loss w.r.t. Kedge_t
%
%  Kedge_t is the edge sensitivity K in c(r) = 1/(1+(r/K)^2).
%  Uses symmetric finite differences.
%
%  NOTE: clean is passed directly (not reconstructed from diff).

h = 1e-4;

p_p             = params;
p_p.Kedge_t(t)  = params.Kedge_t(t) + h;
[u_p, ~]        = forward_stage(u_in_cur, u_in_prev, f, p_p, t, cfg);
L_p             = 0.5 * mean((u_p(:) - clean(:)).^2);

p_m             = params;
p_m.Kedge_t(t)  = max(0.01, params.Kedge_t(t) - h);
[u_m, ~]        = forward_stage(u_in_cur, u_in_prev, f, p_m, t, cfg);
L_m             = 0.5 * mean((u_m(:) - clean(:)).^2);

grad = (L_p - L_m) / (2 * h);
end
