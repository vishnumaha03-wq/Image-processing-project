function visualise_results(res_clean, res_noisy, res_pde, res_tnrd, L)

N_imgs = numel(res_clean);
group_size = 5;

%% Create folder
save_folder = 'results_png';
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

group_counter = 1;

for start_idx = 1:group_size:N_imgs
    
    end_idx = min(start_idx + group_size - 1, N_imgs);
    current_batch = start_idx:end_idx;
    num_batch = length(current_batch);

    %% Create ONE figure for each group
    fig = figure('Visible','on', ...   % change to 'off' if you want no popup at all
        'Color','white', ...
        'Units','pixels', ...
        'Position',[100 50 1400 900], ...
        'Name', sprintf('Group %d', group_counter));

    for i = 1:num_batch
        im = current_batch(i);

        I_clean = res_clean{im};
        I_noisy = res_noisy{im};
        I_pde   = res_pde{im};
        I_tnrd  = res_tnrd{im};

        %% Metrics
        psnr_n = compute_psnr(I_clean, I_noisy);
        psnr_p = compute_psnr(I_clean, I_pde);
        psnr_t = compute_psnr(I_clean, I_tnrd);

        ssim_p = compute_ssim(I_clean, I_pde);
        ssim_t = compute_ssim(I_clean, I_tnrd);

        panels = {I_clean, I_noisy, I_pde, I_tnrd};

        ttls = {
            {'Clean','Reference'}, ...
            {sprintf('Noisy\nPSNR: %.2f dB', psnr_n)}, ...
            {sprintf('PDE\nPSNR: %.2f SSIM: %.4f', psnr_p, ssim_p)}, ...
            {sprintf('TNRD\nPSNR: %.2f SSIM: %.4f', psnr_t, ssim_t)}
        };

        %% Each row = one image (4 columns)
        for k = 1:4
            subplot(num_batch, 4, (i-1)*4 + k);

            imshow(panels{k}, 'InitialMagnification','fit');
            title(ttls{k}, 'FontSize',10);

            axis off;

            % Highlight TNRD result
            if k == 4
                set(gca, 'XColor',[0 0.8 0], ...
                         'YColor',[0 0.8 0], ...
                         'LineWidth',2);
            end
        end
    end

    %% Title
    sgtitle(sprintf('Gamma Noise Removal — L=%d | Group %d', L, group_counter), ...
        'FontSize',14,'FontWeight','bold');

    drawnow;

    %% SAVE PNG (WORKING METHOD)
    filename = fullfile(save_folder, ...
        sprintf('Results_Group_%02d.png', group_counter));

    print(fig, filename, '-dpng', '-r300');   % ✅ stable save

    close(fig);   % ✅ closes window (important)

    group_counter = group_counter + 1;
end

end