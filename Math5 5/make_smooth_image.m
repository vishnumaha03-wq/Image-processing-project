function img = make_smooth_image(N)
%% MAKE_SMOOTH_IMAGE   Synthetic smooth test image for training diversity
%
%  Generates an N x N image with slowly varying sinusoidal intensities,
%  useful as an additional training sample alongside photographic images.

[X, Y] = meshgrid(linspace(0, 1, N), linspace(0, 1, N));
img    = 0.5 + 0.35 * sin(3*pi*X) .* cos(2*pi*Y);
img    = max(0.05, min(0.95, img));
end
