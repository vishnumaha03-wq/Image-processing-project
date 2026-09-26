function [clean_p, noisy_p] = extract_patches(clean_imgs, noisy_imgs, cfg)
%% EXTRACT_PATCHES   Extract aligned patch pairs from image lists
%
%  Outputs:
%    clean_p  — [patch_size x patch_size x 1 x N_patches]
%    noisy_p  — [patch_size x patch_size x 1 x N_patches]

ps = cfg.patch_size;
st = cfg.stride;

clean_list = {};
noisy_list = {};

for im = 1:numel(clean_imgs)
    C = clean_imgs{im};
    N = noisy_imgs{im};
    [H, W] = size(C);
    for r = 1:st:(H - ps + 1)
        for c = 1:st:(W - ps + 1)
            clean_list{end+1} = C(r:r+ps-1, c:c+ps-1);
            noisy_list{end+1} = N(r:r+ps-1, c:c+ps-1);
        end
    end
end

NP      = numel(clean_list);
clean_p = zeros(ps, ps, 1, NP);
noisy_p = zeros(ps, ps, 1, NP);
for i = 1:NP
    clean_p(:,:,1,i) = clean_list{i};
    noisy_p(:,:,1,i) = noisy_list{i};
end
end
