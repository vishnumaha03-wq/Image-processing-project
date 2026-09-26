function run_ablation_study(test_images, params, cfg, L)
%% RUN_ABLATION_STUDY   4-variant ablation following your model design
%
%  Variants tested:
%    1. Full model         — g(u_sigma) * c(|grad u_sigma|), telegraph
%    2. No GLI  (b=1)     — remove gray level indicator
%    3. No Edge (c=1)     — remove edge detector
%    4. Parabolic         — remove telegraph wave/memory term
%
%  Results: table of avg PSNR/SSIM + bar chart

variants = {
    'Full Model',       'full';
    'No GLI  (b=1)',    'no_gray';
    'No Edge (c=1)',    'no_edge';
    'Parabolic',        'parabolic';
};
n_var  = size(variants, 1);
n_imgs = numel(test_images);

psnr_mat = zeros(n_imgs, n_var);
ssim_mat = zeros(n_imgs, n_var);

fprintf('\n  Running ablation variants...\n');

for v = 1:n_var
    mode = variants{v, 2};
    fprintf('    Variant: %s\n', variants{v,1});

    for im = 1:n_imgs
        I_clean = test_images{im};
        I_noisy = add_gamma_noise(I_clean, L);

        I_out = infer_tnrd_telegraph_ablation(I_noisy, params, cfg, mode);

        psnr_mat(im, v) = compute_psnr(I_clean, I_out);
        ssim_mat(im, v) = compute_ssim(I_clean, I_out);
    end
end

%% Print results table
fprintf('\n');
fprintf('  Ablation Study — L=%d\n', L);
fprintf('  %-22s  %10s  %10s\n', 'Variant', 'PSNR (dB)', 'SSIM');
fprintf('  %s\n', repmat('-', 1, 48));
for v = 1:n_var
    fprintf('  %-22s  %10.3f  %10.4f\n', ...
        variants{v,1}, mean(psnr_mat(:,v)), mean(ssim_mat(:,v)));
end
fprintf('  %s\n', repmat('-', 1, 48));

%% Bar chart — PSNR and SSIM side by side
fig = figure('Name', sprintf('Ablation L=%d', L), ...
             'Position', [150 150 1000 420]);

subplot(1, 2, 1);
b1 = bar(mean(psnr_mat, 1), 0.6, 'FaceColor', 'flat');
colors = [0.2 0.5 0.8;   % blue  — full
          0.9 0.4 0.1;   % orange — no GLI
          0.3 0.7 0.3;   % green  — no edge
          0.7 0.2 0.6];  % purple — parabolic
for v = 1:n_var
    b1.CData(v,:) = colors(v,:);
end
set(gca, 'XTick', 1:n_var, 'XTickLabel', variants(:,1), ...
    'XTickLabelRotation', 12, 'FontSize', 9);
ylabel('Average PSNR (dB)', 'FontSize', 10);
title(sprintf('Ablation: PSNR (L=%d)', L), 'FontSize', 11);
grid on; ylim([0, max(mean(psnr_mat,1))*1.15]);

subplot(1, 2, 2);
b2 = bar(mean(ssim_mat, 1), 0.6, 'FaceColor', 'flat');
for v = 1:n_var
    b2.CData(v,:) = colors(v,:);
end
set(gca, 'XTick', 1:n_var, 'XTickLabel', variants(:,1), ...
    'XTickLabelRotation', 12, 'FontSize', 9);
ylabel('Average SSIM', 'FontSize', 10);
title(sprintf('Ablation: SSIM (L=%d)', L), 'FontSize', 11);
grid on; ylim([0, min(1.0, max(mean(ssim_mat,1))*1.15)]);

sgtitle(sprintf('Ablation Study — Telegraph-TNRD, L=%d Gamma Noise', L), ...
        'FontSize', 12, 'FontWeight', 'bold');
end