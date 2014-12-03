function [lassoW, lassoErrors] = testLeastLasso(data)
    %%
    % data should follow the school dataset format. It should also be renormalized
    %
    %% Related functions
    %   mtSplitPerc, CrossValidation4Param_alpha, SolveTreeBased_ElasticNet

    addpath('../../MALSAR/functions/Lasso/');
    addpath('../train_and_test/');
    addpath('../../MALSAR/utils/');

    % load data
    X = data.X;
    Y = data.Y;

    all_trial = 20;
    all_rmse = zeros(3, all_trial);
    % all_perf = zeros(8, all_trial);

    % split data into training and testing.
    training_percent = 0.8;

    for tt = 1:all_trial
        [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);

        % the function used for evaluation.
        eval_func_str = 'eval_MTL_mse';
        higher_better = false;  % mse is lower the better.

        % cross validation fold
        cv_fold = 5;

        % optimization options
        opts = [];
        opts.maxIter = 100;

        % model parameter range
        param_range = [0.1 1 10 100 1000 10000 100000];

        fprintf('Perform model selection via cross validation: \n')
        [ best_param, ~] = CrossValidation1Param...
            ( X_tr, Y_tr, 'Least_Lasso', opts, param_range, ...
            cv_fold, eval_func_str, higher_better);

        % build model using the optimal parameter
        lassoW = Least_Lasso(X_tr, Y_tr, best_param, opts);

        % show final performance
        [f_mse, f_rss, f_tss] = eval_MTL_mse(Y_te, X_te, lassoW);

        all_rmse(:, tt) = [f_mse, f_rss, f_tss];
    end

    lassoErrors = zeros(all_trial, 4);
    lassoErrors(:,1:3) = all_rmse';
    lassoErrors(:,4) = 1 - ( lassoErrors(:, 2) ./ lassoErrors(:, 3) );

    save('lassoW.mat','lassoW');
    save('lassoErrors.mat','lassoErrors');
    save('lassoBest.mat','best_param');
    % save('perf.mat','perform_mat');

