function u_out = infer_tnrd_telegraph_ablation(f, params, cfg, mode)
%% INFER_TNRD_TELEGRAPH_ABLATION   Inference with ablation variant
%
%  mode options:
%    'full'      — full model (no change)
%    'no_gray'   — set b=1 everywhere (remove gray level indicator)
%    'no_edge'   — set c=1 everywhere (remove edge detector)
%    'parabolic' — remove telegraph memory term (gamma -> 0)

f   = double(f);
T   = params.T;
eps = 1e-8;

%% Apply ablation modification to params/cfg
p_abl   = params;
cfg_abl = cfg;

switch lower(mode)
    case 'no_gray'
        % b(s) = 2*s^0/(1+s^0) = 1 for all s when nu -> 0
        % Use very small nu so b ~= 1 everywhere
        for t = 1:T
            p_abl.nu_t(t) = 1e-6;
        end

    case 'no_edge'
        % c = 1/(1+(|g|/K)^2) -> 1 when K -> inf
        for t = 1:T
            p_abl.Kedge_t(t) = 1e8;
        end

    case 'parabolic'
        % Remove wave/memory term by setting gamma = 0
        % With gamma=0: coeff_next=1, coeff_cur=2
        % u^{n+1} = 2*u^n - u^{n-1} + tau^2*div(...) (wave only)
        % To get pure parabolic: also u^{n-1} = u^n
        % => u^{n+1} = u^n + tau^2*div(...)
        cfg_abl.gamma_fix = 0;

    case 'full'
        % No change

    otherwise
        error('Unknown ablation mode: %s', mode);
end

%% Run inference with modified params
avg_lam = mean(p_abl.lambda_t);
if avg_lam > 0.6
    max_cycles = 15;
else
    max_cycles = 10;
end

u_cur  = f;
u_prev = f;

for cycle = 1:max_cycles
    u_before = u_cur;

    for t = 1:T
        [u_cur, u_prev] = forward_stage(u_cur, u_prev, f, p_abl, t, cfg_abl);
    end

    rel_err = norm(u_cur(:) - u_before(:))^2 / ...
              (norm(u_before(:))^2 + eps);
    if rel_err < 1e-4
        break;
    end
end

u_out = max(0, min(1, u_cur));
end