function grad = grad_lambda_stage(u_in_cur, u_in_prev, f, clean, params, t, cfg)
%% GRAD_LAMBDA_STAGE   FD gradient of stage-t loss w.r.t. lambda_t
%
%  Uses symmetric finite differences, consistent with grad_nu_stage,
%  grad_Kedge_stage, and grad_filters_stage.
%
%  NOTE: clean is passed directly (not reconstructed from diff).

h = 1e-4;

p_p = params;
p_p.lambda_t(t) = params.lambda_t(t) + h;
[u_p, ~] = forward_stage(u_in_cur, u_in_prev, f, p_p, t, cfg);
L_p = 0.5 * mean((u_p(:) - clean(:)).^2);

p_m = params;
p_m.lambda_t(t) = max(1e-6, params.lambda_t(t) - h);
[u_m, ~] = forward_stage(u_in_cur, u_in_prev, f, p_m, t, cfg);
L_m = 0.5 * mean((u_m(:) - clean(:)).^2);

grad = (L_p - L_m) / (2 * h);
end