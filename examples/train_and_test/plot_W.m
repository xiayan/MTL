function plot_W(W)
% find the best parameter using cross validation
% for methods contain 1, 2 or 3 parameters

imagesc(W(1:27,:));
colormap('bone');
xlabel('Tasks', 'FontSize', 18);
ylabel('Features', 'FontSize', 18);
colorbar;

