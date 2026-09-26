function params = train_stagewise(params, clean_imgs, noisy_imgs, cfg, L)
%% TRAIN_STAGEWISE   Greedy stage-wise training (TNRD protocol)
%
%  Stage t: freeze params of stages 1..t-1, optimize stage t only
%  Learned per stage: nu_t, Kedge_t, lambda_t, K_t (filter bank)
%  Fixed (phi_i):     gamma, tau, Gaussian kernel, phi_type
%
%  Optimizer: Adam with gradient clipping
%  Loss: MSE between u^{(t)} output and clean image

T      = params.T;
lr     = cfg.lr;
n_imgs = numel(clean_imgs);

fprintf('  Greedy stage-wise training: %d stages, L=%d\n', T, L);

for t = 1:T
    fprintf('\n  ---- Stage %d / %d ----\n', t, T);

    %% Adam states for stage t parameters
    % Scalars
    m_nu  = 0; v_nu  = 0;
    m_Ke  = 0; v_Ke  = 0;
    m_lam = 0; v_lam = 0;
    % Filter bank (fsize x fsize x Nk)
    fs = cfg.fsize; Nk = cfg.Nk;
    m_K = zeros(fs, fs, Nk);
    v_K = zeros(fs, fs, Nk);

    b1 = 0.9; b2 = 0.999; ep_adam = 1e-8;
    grad_clip = 1.0;   % gradient clipping threshold

    best_loss   = Inf;
    best_params = params;
    no_improve  = 0;
    patience    = 80;

    loss_history = zeros(cfg.max_iter, 1);

    for iter = 1:cfg.max_iter

        %% Random mini-batch image
        idx     = randi(n_imgs);
        I_clean = clean_imgs{idx};
        I_noisy = noisy_imgs{idx};

        %% Forward through frozen stages 1..t-1
        u_cur  = I_noisy;
        u_prev = I_noisy;
        for s = 1:t-1
            [u_cur, u_prev] = forward_stage(u_cur, u_prev, ...
                              I_noisy, params, s, cfg);
        end

        %% Compute finite-difference gradients for stage t
        g_nu  = grad_nu_stage(u_cur, u_prev, I_noisy, I_clean, params, t, cfg);
        g_Ke  = grad_Kedge_stage(u_cur, u_prev, I_noisy, I_clean, params, t, cfg);
        g_lam = grad_lambda_stage(u_cur, u_prev, I_noisy, I_clean, params, t, cfg);
        g_K   = grad_filters_stage(u_cur, u_prev, I_noisy, I_clean, params, t, cfg);

        %% Gradient clipping (prevents blow-up)
        g_nu  = clip_scalar(g_nu,  grad_clip);
        g_Ke  = clip_scalar(g_Ke,  grad_clip);
        g_lam = clip_scalar(g_lam, grad_clip);
        g_K   = g_K ./ max(1, max(abs(g_K(:))) / grad_clip);

        %% Adam update — nu_t
        m_nu = b1*m_nu + (1-b1)*g_nu;
        v_nu = b2*v_nu + (1-b2)*g_nu^2;
        mh   = m_nu/(1-b1^iter);
        vh   = v_nu/(1-b2^iter);
        params.nu_t(t) = params.nu_t(t) - lr * mh/(sqrt(vh)+ep_adam);
        params.nu_t(t) = max(0.5, min(5.0, params.nu_t(t)));   % nu in [0.5,5]

        %% Adam update — Kedge_t
        m_Ke = b1*m_Ke + (1-b1)*g_Ke;
        v_Ke = b2*v_Ke + (1-b2)*g_Ke^2;
        mh   = m_Ke/(1-b1^iter);
        vh   = v_Ke/(1-b2^iter);
        params.Kedge_t(t) = params.Kedge_t(t) - lr * mh/(sqrt(vh)+ep_adam);
        params.Kedge_t(t) = max(0.01, min(10.0, params.Kedge_t(t)));  % K > 0

        %% Adam update — lambda_t
        m_lam = b1*m_lam + (1-b1)*g_lam;
        v_lam = b2*v_lam + (1-b2)*g_lam^2;
        mh    = m_lam/(1-b1^iter);
        vh    = v_lam/(1-b2^iter);
        params.lambda_t(t) = params.lambda_t(t) - lr * mh/(sqrt(vh)+ep_adam);
        params.lambda_t(t) = max(0.0, min(2.0, params.lambda_t(t)));  % lambda >= 0

        %% Adam update — filter bank K_t
        m_K = b1*m_K + (1-b1)*g_K;
        v_K = b2*v_K + (1-b2)*g_K.^2;
        mh  = m_K ./ (1-b1^iter);
        vh  = v_K ./ (1-b2^iter);
        params.K_t{t} = params.K_t{t} - lr * mh ./ (sqrt(vh)+ep_adam);

        %% Normalize filters (prevent scale drift, TNRD standard)
        for i = 1:Nk
            fi = params.K_t{t}(:,:,i);
            fi = fi - mean(fi(:));          % zero-mean
            n_ = norm(fi(:));
            if n_ > 1e-8
                params.K_t{t}(:,:,i) = fi / n_;
            end
        end

        %% Compute loss for monitoring
        [u_out, ~] = forward_stage(u_cur, u_prev, I_noisy, params, t, cfg);
        loss = 0.5 * mean((u_out(:) - I_clean(:)).^2);
        loss_history(iter) = loss;

        %% Track best parameters
        if loss < best_loss - 1e-7
            best_loss   = loss;
            best_params = params;
            no_improve  = 0;
        else
            no_improve = no_improve + 1;
        end

        %% Early stopping
        if no_improve >= patience
            fprintf('    Early stop at iter %d\n', iter);
            break;
        end

        %% Logging
        if mod(iter, 100) == 0
            fprintf('    iter %4d | loss=%.6f | nu=%.3f | Ke=%.3f | lam=%.3f\n', ...
                iter, loss, params.nu_t(t), params.Kedge_t(t), params.lambda_t(t));
        end
    end

    %% Restore best params for this stage
    params = best_params;
    fprintf('  Stage %d done | nu=%.3f | Ke=%.3f | lam=%.3f | best_loss=%.6f\n', ...
        t, params.nu_t(t), params.Kedge_t(t), params.lambda_t(t), best_loss);
end

fprintf('\n  Training complete for L=%d\n', L);
end

%% Helper: scalar gradient clipping
function g = clip_scalar(g, threshold)
    if abs(g) > threshold
        g = sign(g) * threshold;
    end
end