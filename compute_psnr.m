function psnr_val = compute_psnr(I_ref, I_test)
%% COMPUTE_PSNR   Peak Signal-to-Noise Ratio (dB)
%  PSNR = 10 * log10( max(I)^2 / MSE )

I_ref  = double(I_ref);
I_test = double(I_test);
mse    = mean((I_ref(:) - I_test(:)).^2);

if mse < 1e-10
    psnr_val = Inf;
else
    psnr_val = 10 * log10(max(I_ref(:))^2 / mse);
end
end
