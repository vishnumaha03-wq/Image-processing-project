function [clean, noisy_L1, noisy_L10] = load_training_data(cfg)
%% LOAD_TRAINING_DATA  Load grayscale training images
%
%  Loads from your 'Dataset' folder first (the ~400 images visible in
%  your MATLAB Drive screenshot), then falls back to MATLAB built-ins,
%  then adds synthetic images as supplement.

bsds_dir = 'Dataset';   % ← your folder shown in the screenshot

clean = {};

%% ── 1. Load from Dataset folder ─────────────────────────────────────────
if isfolder(bsds_dir)
    exts  = {'*.jpg','*.jpeg','*.png','*.tif','*.bmp'};
    files = [];
    for e = 1:numel(exts)
        found = dir(fullfile(bsds_dir, exts{e}));
        files = [files; found]; %#ok<AGROW>
    end
    fprintf('  Found %d files in Dataset/\n', numel(files));

    for i = 1:numel(files)
        try
            img = im2double(imread(fullfile(bsds_dir, files(i).name)));
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            if min(size(img,1), size(img,2)) >= cfg.patch_size
                clean{end+1} = img; %#ok<AGROW>
            end
        catch
        end
        if mod(i, 100) == 0
            fprintf('  Scanned %d files, loaded %d so far ...\n', i, numel(clean));
        end
    end
    fprintf('  Loaded %d images from Dataset/\n', numel(clean));
end

%% ── 2. MATLAB built-ins as fallback if Dataset is empty ─────────────────
if numel(clean) < 5
    fprintf('  Dataset/ empty or missing — using MATLAB built-ins\n');
    builtins = {'cameraman.tif','circuit.tif','coins.png','eight.tif', ...
                'liftingbody.png','moon.tif','pout.tif','rice.png','tire.tif'};
    for i = 1:numel(builtins)
        try
            img = im2double(imread(builtins{i}));
            if size(img, 3) == 3, img = rgb2gray(img); end
            clean{end+1} = img; %#ok<AGROW>
        catch
        end
    end
    fprintf('  Total after built-ins: %d\n', numel(clean));
end

%% ── 3. Always add synthetic images as supplement ─────────────────────────
for sz = [128, 256]
    clean{end+1} = make_circle_image(sz);  %#ok<AGROW>
    clean{end+1} = make_smooth_image(sz);  %#ok<AGROW>
end
fprintf('  Total training images (incl. synthetic): %d\n', numel(clean));

%% ── Apply gamma noise (loop to avoid memory spikes with 400 images) ──────
fprintf('  Generating L=1  noisy versions ...\n');
noisy_L1 = cell(1, numel(clean));
for i = 1:numel(clean)
    noisy_L1{i} = add_gamma_noise(clean{i}, 1);
end

fprintf('  Generating L=10 noisy versions ...\n');
noisy_L10 = cell(1, numel(clean));
for i = 1:numel(clean)
    noisy_L10{i} = add_gamma_noise(clean{i}, 10);
end

fprintf('  Done loading training data.\n');
end