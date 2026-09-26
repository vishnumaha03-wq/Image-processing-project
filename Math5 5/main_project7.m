%% PROJECT 7 — TNRD-Telegraph for Multiplicative Gamma Noise Removal
%  Fixed:   phi_i ('students_t'), gamma
%  Learned: K_t, lambda_t, nu_t, Kedge_t  (stage-wise greedy)
%
%  Reference PDE:  Majee et al., SIAM J. Imaging Sci. 2020
%  TNRD protocol:  Chen & Pock, IEEE TPAMI 2017

clear; clc; close all;

%% ── Hyperparameters ──────────────────────────────────────────────────────
cfg.T          = 5;
cfg.Nk         = 24;
cfg.fsize      = 5;
cfg.gamma_fix  = 2.0;
cfg.tau        = 0.1;
cfg.epsilon    = 1e-6;
cfg.patch_size = 40;
cfg.stride     = 20;
cfg.lr         = 1e-3;
cfg.max_iter   = 800;

%% ── Training data (uses your Dataset folder) ─────────────────────────────
fprintf('=== Loading training data ===\n');
[train_clean, train_noisy_L1, train_noisy_L10] = load_training_data(cfg);

%% ── Stage-wise training ──────────────────────────────────────────────────
fprintf('\n=== Stage-wise Training (L=1) ===\n');
params_L1  = init_params(cfg);
params_L1  = train_stagewise(params_L1,  train_clean, train_noisy_L1,  cfg, 1);

fprintf('\n=== Stage-wise Training (L=10) ===\n');
params_L10 = init_params(cfg);
params_L10 = train_stagewise(params_L10, train_clean, train_noisy_L10, cfg, 10);

%% ── Test images ──────────────────────────────────────────────────────────
fprintf('\n=== Loading test images ===\n');
test_images = load_test_images();
n_test      = numel(test_images);

%% ── Evaluation loop ──────────────────────────────────────────────────────
for L_idx = 1:2
    L_vals = [1, 10];
    L      = L_vals(L_idx);
    if L == 1,  params = params_L1;
    else,       params = params_L10;
    end

    fprintf('\n=== Testing & Comparison (L=%d) ===\n', L);
    fprintf('%-8s %-12s %-12s %-12s %-12s %-12s\n', ...
            'Image','PSNR Noisy','PSNR PDE','PSNR TNRD','SSIM PDE','SSIM TNRD');
    fprintf('%s\n', repmat('-',1,68));

    % Pre-allocate metric arrays
    psnr_noisy_all = zeros(1, n_test);
    psnr_pde_all   = zeros(1, n_test);
    psnr_tnrd_all  = zeros(1, n_test);
    ssim_pde_all   = zeros(1, n_test);
    ssim_tnrd_all  = zeros(1, n_test);

    % Store results for visualisation
    res_clean = cell(1, n_test);
    res_noisy = cell(1, n_test);
    res_pde   = cell(1, n_test);
    res_tnrd  = cell(1, n_test);

    for im = 1:n_test
        I_clean = test_images{im};
        I_noisy = add_gamma_noise(I_clean, L);
        I_tnrd  = infer_tnrd_telegraph(I_noisy, params, cfg);
        I_pde   = run_pde_baseline(I_noisy, L);

        res_clean{im} = I_clean;
        res_noisy{im} = I_noisy;
        res_pde{im}   = I_pde;
        res_tnrd{im}  = I_tnrd;

        psnr_noisy_all(im) = compute_psnr(I_clean, I_noisy);
        psnr_pde_all(im)   = compute_psnr(I_clean, I_pde);
        psnr_tnrd_all(im)  = compute_psnr(I_clean, I_tnrd);
        ssim_pde_all(im)   = compute_ssim(I_clean, I_pde);
        ssim_tnrd_all(im)  = compute_ssim(I_clean, I_tnrd);

        fprintf('%-8d %-12.3f %-12.3f %-12.3f %-12.4f %-12.4f\n', im, ...
                psnr_noisy_all(im), psnr_pde_all(im), psnr_tnrd_all(im), ...
                ssim_pde_all(im),   ssim_tnrd_all(im));
    end

    % Mean row
    fprintf('%s\n', repmat('-',1,68));
    fprintf('%-8s %-12.3f %-12.3f %-12.3f %-12.4f %-12.4f\n', 'MEAN', ...
            mean(psnr_noisy_all), mean(psnr_pde_all), mean(psnr_tnrd_all), ...
            mean(ssim_pde_all),   mean(ssim_tnrd_all));

    %% Visualise — pass pre-computed result arrays (no recomputation)
    visualise_results(res_clean, res_noisy, res_pde, res_tnrd, L);

    %% Ablation study
    fprintf('\n=== Ablation Study (L=%d) ===\n', L);
    run_ablation_study(test_images, params, cfg, L);
end