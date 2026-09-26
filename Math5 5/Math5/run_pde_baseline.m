function I_out = run_pde_baseline(I_noisy, L)
%% RUN_PDE_BASELINE - Stable 5-point finite difference stencil
%  Implemented from Majee et al. 2020 (A Gray Level Indicator-Based 
%  Regularized Telegraph Diffusion Model)
I = double(I_noisy);

switch L
    case 1
        gamma = 0.5;  nu = 1;  K = 0.5;  xi = 1;
    case 10
        gamma = 2.0;  nu = 1;  K = 1.0;  xi = 1;
    otherwise
        gamma = 1.0;  nu = 1;  K = 1.0;  xi = 1;
end

tau      = 0.2;
max_iter = 200;
eps_val  = 1e-6;

I_cur  = I;
I_prev = I;
I_best = I;

for n = 1:max_iter
    I_xi  = imgaussfilt(I_cur, xi);
    M_Ixi = max(I_xi(:)) + eps_val;

    s = I_xi / M_Ixi;
    b = 2 .* s.^nu ./ (1 + s.^nu);

    Ix = I_xi(:, [2:end, end]) - I_xi(:, [1, 1:end-1]);
    Iy = I_xi([2:end, end], :) - I_xi([1, 1:end-1], :);
    
    grad_mag = sqrt(Ix.^2 + Iy.^2 + eps_val);
    c_val    = 1 ./ (1 + (grad_mag / K).^2);
    g        = b .* c_val;

    % Stable 5-point explicit stencil
    I_N = I_cur([1, 1:end-1], :);
    I_S = I_cur([2:end, end], :);
    I_W = I_cur(:, [1, 1:end-1]);
    I_E = I_cur(:, [2:end, end]);

    g_N = (g + g([1, 1:end-1], :)) / 2;
    g_S = (g + g([2:end, end], :)) / 2;
    g_W = (g + g(:, [1, 1:end-1])) / 2;
    g_E = (g + g(:, [2:end, end])) / 2;

    flux_N = g_N .* (I_N - I_cur);
    flux_S = g_S .* (I_S - I_cur);
    flux_W = g_W .* (I_W - I_cur);
    flux_E = g_E .* (I_E - I_cur);

    div_term = flux_N + flux_S + flux_W + flux_E;

    coeff = 1 + gamma * tau;
    I_new = ((2 + gamma*tau)*I_cur - I_prev + tau^2*div_term) / coeff;

    % Neumann boundary conditions
    I_new(1,:)   = I_new(2,:);
    I_new(end,:) = I_new(end-1,:);
    I_new(:,1)   = I_new(:,2);
    I_new(:,end) = I_new(:,end-1);

    I_new  = max(eps_val, I_new);
    I_best = I_new;

    % BUG FIX: Squared L2 norms to match paper's stopping criterion (Eq 4.2)
    rel_err = (norm(I_new(:) - I_cur(:))^2) / (norm(I_cur(:))^2 + eps_val);
    
    if rel_err < 1e-4
        fprintf('    PDE converged at iteration %d\n', n);
        break;
    end
    
    I_prev = I_cur;
    I_cur  = I_new;
end

I_out = max(0, min(1, I_best));
end