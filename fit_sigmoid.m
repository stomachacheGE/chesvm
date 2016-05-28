x = [0.6 0.7 0.55 0.9 1 -0.5 -0.1 -0.9 0.8 -0.4];
y = [1 1 1 1 1 0 0 0 0 0];


sigfunc = @(A, x)(1 ./ (1 + exp(- A(1) .* x + A(2))));
A0 = ones(1,2); %// Initial values fed into the iterative algorithm
A_fit = nlinfit(x, y, sigfunc, A0);

figure(1);
f = @(x)(1 ./ (1 + exp(- A_fit(1) .* x + A_fit(2))));
fplot(f, [-2 2]);
hold on;
scatter(x,y);
hold off;

