
function A_fit = fit_sigmoid(x, y)
% Fit a sigmoid function and display the function curve.
%
% By Liangcheng Fu.
%
% This file is part of the chesvm package, which train exemplar-SVMs using
% HoG and CNN features. Inspired by exemplarsvm from Tomasz Malisiewicz.
% Package homepage: https://github.com/stomachacheGE/chesvm/

show_fit = true;
sigfunc = @(A, x)(1 ./ (1 + exp(- A(1) .* x + A(2))));
A0 = ones(1,2); %// Initial values fed into the iterative algorithm
A_fit = nlinfit(x, y, sigfunc, A0);
if show_fit
    figure(1);
    f = @(x)(1 ./ (1 + exp(- A_fit(1) .* x + A_fit(2))));

    fplot(f, [min(x) max(x)]);
    hold on;
    scatter(x,y);
    hold off;
end
end