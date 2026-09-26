function out = apply_phi(r, phi_type)
%% APPLY_PHI   Fixed influence function phi_i applied to filter response r
%
%  phi_i is the derivative of a robust potential rho_i.
%  FIXED (not learned) in Project 7.
%
%  Types:
%    'half_quadratic' : phi(r) = r / sqrt(r^2 + alpha^2)
%    'students_t'     : phi(r) = 2r / (alpha^2 + r^2)
%    'lorentzian'     : phi(r) = r / (1 + (r/alpha)^2)
%    'perona_malik'   : phi(r) = r * exp(-r^2 / (2*alpha^2))
%    'identity'       : phi(r) = r  (Tikhonov — for ablation)

alpha = 1.0;

switch lower(phi_type)
    case 'half_quadratic'
        out = r ./ sqrt(r.^2 + alpha^2);

    case 'students_t'
        out = 2 .* r ./ (alpha^2 + r.^2);

    case 'lorentzian'
        out = r ./ (1 + (r / alpha).^2);

    case 'perona_malik'
        out = r .* exp(-r.^2 / (2 * alpha^2));

    case 'identity'
        out = r;

    otherwise
        error('apply_phi: unknown phi_type ''%s''', phi_type);
end
end
