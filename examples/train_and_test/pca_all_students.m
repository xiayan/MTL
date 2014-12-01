function pca_all_students(X, Y)
% find the best parameter using cross validation
% for methods contain 1, 2 or 3 parameters

coeff = pca(X');

idx1 = (Y < 20);
idx2 = (Y < 40) & (Y >= 20);
idx3 = (Y >= 40);

hold on;
scatter(coeff(idx1,1), coeff(idx1,2), [], [1,0,0]);
scatter(coeff(idx2,1), coeff(idx2,2), [], [0,1,0]);
scatter(coeff(idx3,1), coeff(idx3,2), [], [0,0,1]);

xlabel('Dimension 1', 'FontSize', 18);
ylabel('Dimension 2', 'FontSize', 18);
