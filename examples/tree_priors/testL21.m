function [l21W, l21Errors] = testL21(data)
    %%
    % data should follow the school dataset format. It should also be renormalized
    %
    %% Related functions
    %   mtSplitPerc, CrossValidation4Param_alpha, SolveTreeBased_ElasticNet

    addpath('../../MALSAR/functions/joint_feature_learning/');
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
            ( X_tr, Y_tr, 'Least_L21', opts, param_range, ...
            cv_fold, eval_func_str, higher_better);

        % build model using the optimal parameter
        l21W = Least_L21(X_tr, Y_tr, best_param, opts);

        % show final performance
        [f_mse, f_rss, f_tss] = eval_MTL_mse(Y_te, X_te, l21W);

        all_rmse(:, tt) = [f_mse, f_rss, f_tss];
    end

    l21Errors = zeros(all_trial, 4);
    l21Errors(:,1:3) = all_rmse';
    l21Errors(:,4) = 1 - ( l21Errors(:, 2) ./ l21Errors(:, 3) );

    save('l21W.mat','l21W');
    save('l21Errors.mat','l21Errors');
    save('l21Best.mat','best_param');
    % save('perf.mat','perform_mat');

