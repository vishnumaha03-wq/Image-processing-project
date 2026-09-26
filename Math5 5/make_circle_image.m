function img = make_circle_image(N)
%% MAKE_CIRCLE_IMAGE   Synthetic circle test image as in Majee et al. Fig 1c
%
%  N x N image with:
%    background       = 0.3
%    outer circle     = 0.7
%    inner square     = 0.4

[X, Y] = meshgrid(1:N, 1:N);
cx = N/2;  cy = N/2;

img = 0.3 * ones(N, N);

mask_outer = ((X-cx).^2 + (Y-cy).^2) <= (0.4*N)^2;
img(mask_outer) = 0.7;

sq = round(N*0.15);
r1 = round(cy - sq);  r2 = round(cy + sq);
c1 = round(cx - sq);  c2 = round(cx + sq);
r1 = max(1,r1); r2 = min(N,r2);
c1 = max(1,c1); c2 = min(N,c2);
img(r1:r2, c1:c2) = 0.4;
end
