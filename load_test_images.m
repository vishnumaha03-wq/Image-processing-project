function [test_imgs, names] = load_test_images(test_dir)
%% LOAD_TEST_IMAGES   Load BSD68 test images (no download needed)
%
%  If test_dir is not supplied, automatically searches for your BSD68
%  folder under several common names and locations.
%  Accepts .png, .jpg, .jpeg, .tif, .bmp — whichever your folder has.

%% ── Locate the folder ────────────────────────────────────────────────────
if nargin < 1 || isempty(test_dir)
    % Try common names — stops at the first one that exists and has images
    candidates = { ...
        'bsd68', 'BSD68', 'bsd_68', 'BSD_68', ...
        'test_images', 'test', 'Test', ...
        'BSD68_images', 'bsd68_images', ...
        fullfile('..', 'bsd68'), fullfile('..', 'BSD68') };

    test_dir = '';
    for k = 1:numel(candidates)
        if isfolder(candidates{k})
            % Check it actually contains image files
            found = count_images(candidates{k});
            if found > 0
                test_dir = candidates{k};
                fprintf('  Found BSD68 folder: ./%s/  (%d images)\n', ...
                        test_dir, found);
                break;
            end
        end
    end

    if isempty(test_dir)
        error(['Could not find your BSD68 folder.\n' ...
               'Please call:  load_test_images(''your_folder_name'')']);
    end
end

%% ── Collect image files (all common extensions) ──────────────────────────
exts  = {'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
files = [];
for e = 1:numel(exts)
    files = [files; dir(fullfile(test_dir, exts{e}))]; %#ok<AGROW>
end

if isempty(files)
    error('Folder "%s" exists but contains no image files.', test_dir);
end

% Sort by filename so order is deterministic
[~, order] = sort({files.name});
files      = files(order);

%% ── Load images ─────────────────────────────────────────────────────────
test_imgs = {};
names     = {};

for i = 1:numel(files)
    fpath = fullfile(test_dir, files(i).name);
    try
        img = im2double(imread(fpath));
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        test_imgs{end+1} = img;                         %#ok<AGROW>
        names{end+1}     = files(i).name;               %#ok<AGROW>
    catch e
        fprintf('  Warning: could not load %s (%s)\n', files(i).name, e.message);
    end
end

fprintf('  Loaded %d test images from ./%s/\n', numel(test_imgs), test_dir);
end


%% ── Count image files in a folder (helper) ───────────────────────────────
function n = count_images(folder)
exts = {'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
n    = 0;
for e = 1:numel(exts)
    n = n + numel(dir(fullfile(folder, exts{e})));
end
end