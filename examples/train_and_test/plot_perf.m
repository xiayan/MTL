function plot_perf(perform_mat)

param_range = [0.001 0.01 0.1 1 10 100 1000 10000];
plot(log10(param_range), perform_mat, ':o');
xlabel('log lambda', 'FontSize', 18);
ylabel('MSE', 'FontSize', 18);

