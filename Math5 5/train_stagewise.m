function params = train_stagewise(params, clean_imgs, noisy_imgs, cfg, L)
%% TRAIN_STAGEWISE  Greedy stage-wise training — fully corrected.
%
%  BUGS FIXED vs previous versions:
%
%  BUG 1 — 4-D arrays in forward_stage:
%    train_stagewise was passing [ps×ps×1×N_batch] 4-D arrays to
%    forward_stage.  gradient(u_xi) on a 4-D array computes derivatives
%    along ALL dimensions, not just x and y.  This makes g_w garbage,
%    which corrupts ALL subsequent gradients (filter and FD).
%    FIX: store pre-propagated patches as cell arrays of 2-D images.
%         forward_stage is called with 2-D patches only.
%
%  BUG 2 — FD step h=1e-4 too small for lambda/nu/Ke:
%    Effective change in u from Δlam=1e-4 is τ²·Δlam·F ≈ 0.01·1e-4·10 = 1e-5.
%    This is below the MSE loss's numerical noise floor (~1e-4).
%    FD gradient ≈ (noise)/(2·1e-4) ≈ random sign → Adam diverges.
%    FIX: use h_scalar = 0.05 (500× larger) for lam/nu/Ke.
%         Effective Δu ≈ 0.01·0.05·10 = 0.005 → 5% of pixel range → detectable.
%
%  BUG 3 — lam lower bound 1e-4:
%    With bug 1+2, lam gradient had random sign → lam drifted to lower bound.
%    FIX: lower bound = 0.05 (not 1e-4).
%
%  PROTOCOL (TNRD supplemental §2.1):
%    For stage t: propagate ALL patches through fixed stages 1..t-1 once,
%    store as buf_cur/buf_prev.  Train stage t on these fixed inputs.
%    After training, propagate again to produce buf_cur for stage t+1.

set(0,'DefaultFigureVisible','off');
fprintf('  Stage-wise training for L=%d ...\n', L);

T        = params.T;
Nk       = params.Nk;
fs       = params.fsize;
max_iter = cfg.max_iter;

%% ── Extract patches (4-D) then convert to cell arrays of 2-D ─────────────
fprintf('  Extracting patches...\n');
[clean_p4, noisy_p4] = extract_patches(clean_imgs, noisy_imgs, cfg);
NP = size(noisy_p4, 4);
fprintf('  Total patches: %d\n', NP);

%  Convert to cell arrays of 2-D for clean 2-D processing
noisy_cell = cell(1, NP);
clean_cell = cell(1, NP);
for b = 1:NP
    noisy_cell{b} = squeeze(noisy_p4(:,:,1,b));
    clean_cell{b} = squeeze(clean_p4(:,:,1,b));
end
clear noisy_p4 clean_p4;   % free 4-D memory

%% ── Two-step telegraph buffers (cell arrays of 2-D) ─────────────────────
%  buf_cur{b}  = u^{t-1} for patch b  (input to stage t)
%  buf_prev{b} = u^{t-2} for patch b  (telegraph memory for stage t)
%  Initial: buf_cur = buf_prev = noisy (encodes I_t(x,0)=0)
buf_cur  = noisy_cell;
buf_prev = noisy_cell;

%% ── FD step sizes ────────────────────────────────────────────────────────
h_scalar = 0.05;   % lambda, nu, Ke  (large enough to produce detectable Δu)

%% ── Adam hyperparameters ─────────────────────────────────────────────────
beta1    = 0.9;
beta2    = 0.999;
adam_eps = 1e-8;
lr_K     = cfg.lr;          % filter lr
lr_s     = cfg.lr * 20;     % scalar lr (larger because FD grads are smaller)
B        = min(48, NP);     % mini-batch size
patience = 80;              % early-stop patience

all_loss = zeros(T, max_iter);

for t = 1:T
    fprintf('\n  --- Stage %d / %d ---\n', t, T);

    %% Warm-start scalars from previous stage
    if t > 1
        params.lambda_t(t) = params.lambda_t(t-1);
        params.nu_t(t)     = params.nu_t(t-1);
        params.Kedge_t(t)  = params.Kedge_t(t-1);
    end

    %% Adam states
    mK=zeros(fs,fs,Nk); vK=zeros(fs,fs,Nk);
    m_lam=0; v_lam=0;
    m_nu =0; v_nu =0;
    m_Ke =0; v_Ke =0;
    adam_step = 0;

    best_loss  = Inf;
    best_snap  = snap(params,t);
    no_improve = 0;
    loss_hist  = zeros(1, max_iter);

    for iter = 1:max_iter
        adam_step = adam_step + 1;

        %% Sample mini-batch
        idx = randperm(NP, B);

        %% Accumulate gradients over mini-batch (each patch is 2-D)
        gK_sum   = zeros(fs, fs, Nk);
        glam_sum = 0;  gnu_sum = 0;  gKe_sum = 0;
        loss_sum = 0;

        for b_i = 1:B
            bi  = idx(b_i);
            uc  = buf_cur{bi};      % 2-D [ps × ps]
            up  = buf_prev{bi};     % 2-D
            fb  = noisy_cell{bi};   % 2-D
            gt  = clean_cell{bi};   % 2-D

            %% Forward
            [u_out, ~] = forward_stage(uc, up, fb, params, t, cfg);
            loss_sum = loss_sum + 0.5*mean((u_out(:)-gt(:)).^2);

            %% Filter gradient (analytic)
            gK_sum = gK_sum + grad_filters_stage(uc, up, fb, gt, params, t, cfg);

            %% Scalar gradients via symmetric FD (h=0.05 — large enough to work)
            % lambda
            pp=params; pp.lambda_t(t)=params.lambda_t(t)+h_scalar;
            pm=params; pm.lambda_t(t)=max(0.01,params.lambda_t(t)-h_scalar);
            [up_,~]=forward_stage(uc,up,fb,pp,t,cfg);
            [um_,~]=forward_stage(uc,up,fb,pm,t,cfg);
            glam_sum = glam_sum + ...
                (0.5*mean((up_(:)-gt(:)).^2) - 0.5*mean((um_(:)-gt(:)).^2))/(2*h_scalar);

            % nu
            pp=params; pp.nu_t(t)=params.nu_t(t)+h_scalar;
            pm=params; pm.nu_t(t)=max(0.1,params.nu_t(t)-h_scalar);
            [up_,~]=forward_stage(uc,up,fb,pp,t,cfg);
            [um_,~]=forward_stage(uc,up,fb,pm,t,cfg);
            gnu_sum = gnu_sum + ...
                (0.5*mean((up_(:)-gt(:)).^2) - 0.5*mean((um_(:)-gt(:)).^2))/(2*h_scalar);

            % Ke
            pp=params; pp.Kedge_t(t)=params.Kedge_t(t)+h_scalar;
            pm=params; pm.Kedge_t(t)=max(0.05,params.Kedge_t(t)-h_scalar);
            [up_,~]=forward_stage(uc,up,fb,pp,t,cfg);
            [um_,~]=forward_stage(uc,up,fb,pm,t,cfg);
            gKe_sum = gKe_sum + ...
                (0.5*mean((up_(:)-gt(:)).^2) - 0.5*mean((um_(:)-gt(:)).^2))/(2*h_scalar);
        end

        gK   = gK_sum   / B;
        glam = glam_sum / B;
        gnu  = gnu_sum  / B;
        gKe  = gKe_sum  / B;
        avg_loss = loss_sum / B;
        loss_hist(iter) = avg_loss;

        %% Adam — filters
        mK = beta1*mK + (1-beta1)*gK;
        vK = beta2*vK + (1-beta2)*gK.^2;
        mK_h = mK/(1-beta1^adam_step);
        vK_h = vK/(1-beta2^adam_step);
        K_new = params.K_t{t} - lr_K*mK_h./(sqrt(vK_h)+adam_eps);
        params.K_t{t} = proj_filters(K_new, fs, Nk);

        %% Adam — lambda  [bounds: 0.05 … 5.0]
        m_lam=beta1*m_lam+(1-beta1)*glam; v_lam=beta2*v_lam+(1-beta2)*glam^2;
        dlam = (m_lam/(1-beta1^adam_step)) / (sqrt(v_lam/(1-beta2^adam_step))+adam_eps);
        params.lambda_t(t) = max(0.05, min(5.0, params.lambda_t(t) - lr_s*dlam));

        %% Adam — nu  [bounds: 0.3 … 3.0]
        m_nu=beta1*m_nu+(1-beta1)*gnu; v_nu=beta2*v_nu+(1-beta2)*gnu^2;
        dnu = (m_nu/(1-beta1^adam_step)) / (sqrt(v_nu/(1-beta2^adam_step))+adam_eps);
        params.nu_t(t) = max(0.3, min(3.0, params.nu_t(t) - lr_s*dnu));

        %% Adam — Ke  [bounds: 0.1 … 8.0]
        m_Ke=beta1*m_Ke+(1-beta1)*gKe; v_Ke=beta2*v_Ke+(1-beta2)*gKe^2;
        dKe = (m_Ke/(1-beta1^adam_step)) / (sqrt(v_Ke/(1-beta2^adam_step))+adam_eps);
        params.Kedge_t(t) = max(0.1, min(8.0, params.Kedge_t(t) - lr_s*dKe));

        %% Track best
        if avg_loss < best_loss - 1e-7
            best_loss = avg_loss;
            best_snap = snap(params, t);
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if mod(iter,50)==0
            fprintf('    iter %3d | loss=%.6f | lam=%.4f | nu=%.4f | Ke=%.4f\n', ...
                iter, avg_loss, params.lambda_t(t), params.nu_t(t), params.Kedge_t(t));
        end
        if no_improve >= patience
            fprintf('   Early stop at iter %d\n', iter);
            break;
        end
    end

    params = restore(params, t, best_snap);
    all_loss(t,:) = loss_hist;
    fprintf('  Stage %d done | lam=%.4f | nu=%.4f | Ke=%.4f | best_loss=%.6f\n', ...
        t, params.lambda_t(t), params.nu_t(t), params.Kedge_t(t), best_loss);

    %% ── Propagate ALL patches through newly-trained stage t ──────────────
    %  This gives the correct input distribution for stage t+1.
    %  TNRD supplemental §2.1: "images u_{(t-1)p} are fixed, served as input"
    fprintf('  Propagating %d patches for stage %d input...\n', NP, t+1);
    buf_next = cell(1, NP);
    chunk = 500;
    for s0 = 1:chunk:NP
        e0 = min(s0+chunk-1, NP);
        for bi = s0:e0
            [u_out_b, ~] = forward_stage(buf_cur{bi}, buf_prev{bi}, ...
                                          noisy_cell{bi}, params, t, cfg);
            buf_next{bi} = max(0, min(1, u_out_b));
        end
        if mod(e0, 5000) < chunk
            fprintf('    propagated %d / %d\n', e0, NP);
        end
    end
    buf_prev = buf_cur;
    buf_cur  = buf_next;
    fprintf('  Propagation done.\n');
end

%% Joint fine-tuning (if requested)
if isfield(cfg,'joint_iters') && cfg.joint_iters > 0
    params = joint_finetune(params, clean_cell, noisy_cell, cfg, L);
end

save_loss_plot(all_loss, T, max_iter, L);
set(0,'DefaultFigureVisible','on');
fprintf('\n  Training complete for L=%d\n', L);
end


%% ─── Joint fine-tuning (lambda only, all stages simultaneously) ───────────
function params = joint_finetune(params, clean_cell, noisy_cell, cfg, L)
fprintf('\n  --- Joint Fine-Tuning (%d iters) ---\n', cfg.joint_iters);
T  = params.T;
NP = numel(noisy_cell);
lr = cfg.lr * 0.05;    % 20× smaller for joint phase

best_loss = Inf;
best_lam  = params.lambda_t;

for iter = 1:cfg.joint_iters
    B   = min(32, NP);
    idx = randperm(NP, B);
    loss_sum = 0;  glam_sum = zeros(1,T);

    h_joint = 0.05;

    for b_i = 1:B
        bi = idx(b_i);
        fb = noisy_cell{bi};
        gt = clean_cell{bi};

        % Full T-stage forward
        uc = fb; up = fb;
        for s = 1:T, [uc,up]=forward_stage(uc,up,fb,params,s,cfg); end
        loss_sum = loss_sum + 0.5*mean((uc(:)-gt(:)).^2);

        % Gradient per stage lambda
        for s = 1:T
            pp=params; pp.lambda_t(s)=params.lambda_t(s)+h_joint;
            pm=params; pm.lambda_t(s)=max(0.05,params.lambda_t(s)-h_joint);
            uc_p=fb;up_p=fb; for si=1:T,[uc_p,up_p]=forward_stage(uc_p,up_p,fb,pp,si,cfg);end
            uc_m=fb;up_m=fb; for si=1:T,[uc_m,up_m]=forward_stage(uc_m,up_m,fb,pm,si,cfg);end
            glam_sum(s) = glam_sum(s) + ...
                (0.5*mean((uc_p(:)-gt(:)).^2)-0.5*mean((uc_m(:)-gt(:)).^2))/(2*h_joint);
        end
    end
    loss_avg = loss_sum/B;
    for s=1:T
        params.lambda_t(s) = max(0.05, min(5, ...
            params.lambda_t(s) - lr*sign(glam_sum(s)/B)*min(abs(glam_sum(s)/B),1)));
    end
    if loss_avg < best_loss
        best_loss = loss_avg;
        best_lam  = params.lambda_t;
    end
    if mod(iter,50)==0
        fprintf('    Joint iter %3d | loss=%.6f\n', iter, loss_avg);
    end
end
params.lambda_t = best_lam;
fprintf('  Joint tuning done | best_loss=%.6f\n', best_loss);
end


%% ─── Helpers ─────────────────────────────────────────────────────────────
function sp = snap(params, t)
sp.K   = params.K_t{t};
sp.lam = params.lambda_t(t);
sp.nu  = params.nu_t(t);
sp.Ke  = params.Kedge_t(t);
end

function params = restore(params, t, sp)
params.K_t{t}      = sp.K;
params.lambda_t(t) = sp.lam;
params.nu_t(t)     = sp.nu;
params.Kedge_t(t)  = sp.Ke;
end

function Ko = proj_filters(Ki, fs, Nk)
Ko = Ki;
for i = 1:Nk
    f = Ki(:,:,i); f=f-mean(f(:)); n=norm(f(:));
    if n>1e-8, f=f/n; end
    Ko(:,:,i) = f;
end
end

function save_loss_plot(all_loss, T, max_iter, L)
fig    = figure('Visible','off','Position',[100 100 900 450]);
colors = lines(T);
hold on;
for t = 1:T
    iters = find(all_loss(t,:)>0);
    if ~isempty(iters)
        plot(iters, all_loss(t,iters), '-', 'Color', colors(t,:), ...
             'LineWidth', 2, 'DisplayName', sprintf('Stage %d',t));
    end
end
hold off;
xlabel('Iteration','FontSize',13); ylabel('MSE Loss','FontSize',13);
title(sprintf('Stage-wise Training Loss (L=%d)',L),'FontSize',14,'FontWeight','bold');
legend('Location','northeast','FontSize',11); grid on;
fname = sprintf('loss_curves_L%d.png',L);
print(fig, fname, '-dpng', '-r150'); close(fig);
fprintf('  Loss curve saved: %s\n', fname);
end