k = 1000;
n = 100;
d = 2;
s = .5;

x = randn(d,n);
g = [linspace(-5,5,1000);linspace(-5,5,1000)];

p = kde_test(kde_train(x,s,k),g);

plot(g,p);

disp(['Integral: ' num2str(trapz(g(2,:),p))]);
