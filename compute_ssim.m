function ssim_val = compute_ssim(I_ref, I_test)
%% COMPUTE_SSIM   Structural Similarity Index
%  Uses MATLAB built-in ssim() if available, else simplified global SSIM.

I_ref  = double(I_ref);
I_test = double(I_test);

try
    ssim_val = ssim(I_test, I_ref);
catch
    mu1  = mean(I_ref(:));   mu2 = mean(I_test(:));
    s1   = std(I_ref(:));    s2  = std(I_test(:));
    s12  = mean((I_ref(:)-mu1).*(I_test(:)-mu2));
    c1   = (0.01)^2;         c2  = (0.03)^2;
    ssim_val = (2*mu1*mu2+c1)*(2*s12+c2) / ...
               ((mu1^2+mu2^2+c1)*(s1^2+s2^2+c2));
end
end
