function I_out = run_pde_baseline(I_noisy, L)
%% RUN_PDE_BASELINE   Majee et al. 2020 telegraph PDE (h=0, no fidelity).
%
%  Equation (2.5-2.7):
%    I_tt + γ I_t = div( g(I_ξ, |∇I_ξ|) ∇I ),
%    ∂_n I = 0  (Neumann),  I(x,0)=I_noisy,  I_t(x,0)=0
%
%  Diffusion coefficient  g = b · c:
%    b(I_ξ) = 2s^ν/(1+s^ν),  s = I_ξ/M_ξ   (gray-level indicator)
%    c(|∇I_ξ|) = 1/(1+(|∇I_ξ|/K)²)          (edge detector)
%
%  Explicit telegraph update (paper p.854-855):
%    (1+γτ) I^{n+1} = (2+γτ) I^n − I^{n-1} + τ² div(g^n ∇I^n)
%
%  ── FINITE-VOLUME DIVERGENCE (stable, consistent, Neumann BC) ────────────
%  Cell-centred stencil with arithmetic inter-cell conductance:
%    div(g ∇I)_{i,j} = g_{E}(I_{i,j+1}-I_{i,j}) − g_{W}(I_{i,j}-I_{i,j-1})
%                    + g_{S}(I_{i+1,j}-I_{i,j}) − g_{N}(I_{i,j}-I_{i-1,j})
%  where g_{E} = (g_{i,j}+g_{i,j+1})/2 etc.
%  Boundary: g = 0 at ghost cells → zero normal flux (Neumann BC) automatic.
%
%  This replaces the broken mixed central/backward stencil that produced
%  checkerboard artifacts and SSIM≈0.005.
%
%  ── PARAMETERS from paper Table 2 (Boat image, representative for BSD68) ─
%  L=1:  γ=5, ν=1, K=2, τ=0.2   (Majee Table 2, Boat L=1)
%  L=10: γ=2, ν=2, K=1, τ=0.2   (Majee Table 2, Boat L=10)

switch L
    case 1
        gamma_pde=5.0; nu=1.0; K=2.0; tau=0.2; max_iter=500;
    case 10
        gamma_pde=2.0; nu=2.0; K=1.0; tau=0.2; max_iter=300;
    case 3
        gamma_pde=4.0; nu=1.5; K=2.0; tau=0.2; max_iter=400;
    case 5
        gamma_pde=2.0; nu=1.5; K=1.0; tau=0.2; max_iter=400;
    otherwise
        gamma_pde=max(1,0.5*sqrt(L)); nu=1.0; K=max(0.5,1/sqrt(L));
        tau=0.2; max_iter=400;
end

xi       = 1.0;
eps_stop = 1e-4;
eps_num  = 1e-8;

I0 = max(eps_num, min(1, double(I_noisy)));
I_prev = I0;
I_curr = I0;

for iter = 1:max_iter
    %% Gaussian smoothing
    I_xi = imgaussfilt(I_curr, xi);
    M_xi = max(I_xi(:)) + eps_num;

    %% Gray-level indicator b
    s    = max(eps_num, min(1-eps_num, I_xi/M_xi));
    b    = 2*(s.^nu)./(1+s.^nu);

    %% Edge detector c
    [gy, gx] = gradient(I_xi);
    c = 1./(1+((sqrt(gx.^2+gy.^2+eps_num))./K).^2);

    %% Diffusion coefficient g = b·c
    g = b .* c;

    %% Finite-volume divergence div(g ∇I) — stable, Neumann BC automatic
    div_gI = fv_div(I_curr, g);

    %% Telegraph update (no fidelity: h=0 in paper)
    I_next = ((2+gamma_pde*tau)*I_curr - I_prev + tau^2*div_gI) ...
             / (1+gamma_pde*tau);
    I_next = max(eps_num, min(1.0, I_next));

    %% Stopping criterion (paper eq.4.2)
    rel_err = sum((I_next(:)-I_curr(:)).^2)/(sum(I_curr(:).^2)+eps_num);
    if rel_err <= eps_stop
        fprintf('  PDE converged at iter %d\n', iter);
        break;
    end

    I_prev = I_curr;
    I_curr = I_next;
end

I_out = I_curr;
end


%% ── Finite-volume divergence ─────────────────────────────────────────────
function divF = fv_div(I, g)
%FV_DIV  Cell-centred finite-volume div(g ∇I) with Neumann BC.
%
%  Inter-cell conductance (arithmetic mean):
%    g_{i,j+½} = (g_{i,j} + g_{i,j+1}) / 2
%
%  Neumann BC: pad g with ZEROS at boundary → zero flux leaving domain.
%  Pad I with REPLICATED values → ghost point equals boundary value
%  (consistent with ∂_n I = 0 via backward difference at boundary).

[M, N] = size(I);

%% x-direction (columns): g padded with 0; I padded with replication
g_xp = [zeros(M,1), g, zeros(M,1)];      % M×(N+2): zero-flux BC
I_xp = [I(:,1),     I, I(:,N)  ];        % M×(N+2): replicated

g_E = (g_xp(:,2:N+1)+g_xp(:,3:N+2))/2;   % conductance at east face  [j+½]
g_W = (g_xp(:,1:N)  +g_xp(:,2:N+1))/2;   % conductance at west face  [j-½]

divFx = g_E.*(I_xp(:,3:N+2)-I_xp(:,2:N+1)) ...
      - g_W.*(I_xp(:,2:N+1)-I_xp(:,1:N));

%% y-direction (rows): same approach
g_yp = [zeros(1,N); g; zeros(1,N)];
I_yp = [I(1,:);     I; I(M,:)   ];

g_S = (g_yp(2:M+1,:)+g_yp(3:M+2,:))/2;   % south face [i+½]
g_N = (g_yp(1:M,  :)+g_yp(2:M+1,:))/2;   % north face [i-½]

divFy = g_S.*(I_yp(3:M+2,:)-I_yp(2:M+1,:)) ...
      - g_N.*(I_yp(2:M+1,:)-I_yp(1:M,  :));

divF = divFx + divFy;
end