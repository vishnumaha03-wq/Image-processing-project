function u_out = infer_tnrd_telegraph(f, params, cfg)
%% INFER_TNRD_TELEGRAPH   Multi-stage inference for your TNRD-Telegraph model
%
%  Runs T learned stages repeatedly until convergence.
%  Stopping: relative error < 1e-4 (same as paper eq. 4.2)
%
%  Strategy:
%    - Cycle through stages 1..T repeatedly
%    - Stop when relative change is small enough
%    - Cap at max_cycles to prevent over-smoothing

f       = double(f);
T       = params.T;
eps_val = 1e-8;

%% Adaptive number of cycles based on learned lambda
%  Heavy noise (L=1): lambda_t learned larger -> more iterations needed
%  Light noise (L=10): lambda_t smaller -> fewer iterations
avg_lam = mean(params.lambda_t);

if avg_lam > 0.6
    max_cycles = 15;    % L=1: heavy noise, more denoising cycles
else
    max_cycles = 10;    % L=10: mild noise, fewer cycles
end

u_cur  = f;
u_prev = f;    % I^{n-1} = I^0 (zero initial velocity: I_t(x,0)=0)

for cycle = 1:max_cycles
    u_before_cycle = u_cur;

    %% One full pass through all T stages
    for t = 1:T
        [u_cur, u_prev] = forward_stage(u_cur, u_prev, f, params, t, cfg);
    end

    %% Convergence check: relative error across full cycle
    num     = norm(u_cur(:) - u_before_cycle(:), 'fro')^2;
    den     = norm(u_before_cycle(:), 'fro')^2 + eps_val;
    rel_err = num / den;

    if rel_err < 1e-4
        break;
    end
end

u_out = max(0, min(1, u_cur));
end