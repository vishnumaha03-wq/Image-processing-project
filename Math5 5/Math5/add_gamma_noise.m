function I_noisy = add_gamma_noise(I_clean, L)
%% ADD_GAMMA_NOISE   Multiplicative gamma noise with L looks
%
%  Model: J = I * eta,  eta ~ Gamma(L, 1/L)
%  E[eta]=1, Var[eta]=1/L. Larger L = less noise.

[M, N]  = size(I_clean);
eta     = gamrnd(L, 1/L, M, N);
I_noisy = max(1e-6, I_clean .* eta);
end
