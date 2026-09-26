function out = conv2d_batch(img, kernel, shape)
%% CONV2D_BATCH - Fast N-Dimensional Convolution
    out = convn(img, kernel, shape);
end