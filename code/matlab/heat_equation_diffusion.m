function [ Iout ] = heat_equation_diffusion( Iin, dt, fixed_points, maxiter)
% Perform 2D linear heat equation on a given grayscale image
% Input:
%   Iin: input grayscale image
%   dt: \Delta t used in the updating scheme
%   fixed_points: bool matrix which specify the fixed points that do not 
%       diffuse iteratively
%   maxiter: maximum number of iterations
% Output:
%   Iout: output diffused grayscale image

% Function starts here
Iold = double(Iin);
I = zeros(size(Iin));
stop = false;

if dt > .5
    disp("CFL condition not satisfied, dt must be less than 0.5");
    Iout = Iin;
    return;
end

count = 0;
while(~stop)
    I_px = circshift(Iold, [1, 0]);
    I_mx = circshift(Iold, [-1, 0]);
    I_py = circshift(Iold, [0, 1]);
    I_my = circshift(Iold, [0, -1]);
    I = Iold + dt * (I_px + I_mx + I_py + I_my - 4 * Iold); % update formula
    I(fixed_points) = Iold(fixed_points);
    
    count = count + 1;
    if count > maxiter
        break;
    end
    
    if(norm(I - Iold) < 1e-8)
        stop = true;
    end
    Iold = I;
    imshow(uint8(I));
end

Iout = I;

end

