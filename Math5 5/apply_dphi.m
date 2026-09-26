function dphi = apply_dphi(r, phi_type)
%% APPLY_DPHI  Derivative of influence function phi(r)
alpha = 1.0;
switch lower(phi_type)
    case 'half_quadratic'
        dphi = alpha^2 ./ (r.^2 + alpha^2).^(1.5);
    case 'students_t'
        dphi = 2*(alpha^2 - r.^2) ./ (alpha^2 + r.^2).^2;
    case 'lorentzian'
        dphi = (1 - (r/alpha).^2) ./ (1 + (r/alpha).^2).^2;
    case 'perona_malik'
        dphi = (1 - r.^2/alpha^2) .* exp(-r.^2/(2*alpha^2));
    otherwise
        dphi = alpha^2 ./ (r.^2 + alpha^2).^(1.5);
end
end