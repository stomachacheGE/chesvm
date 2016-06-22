
function A_fit = fit_sigmoid(x, y)
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