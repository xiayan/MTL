function [fixedW, fixedErrors, best_param] = testFixedTree(data, Clusters)
    %%
    % data should follow the school dataset format. It should also be renormalized
    %
    %% Related functions
    %   mtSplitPerc, CrossValidation4Param_alpha, SolveTreeBased_ElasticNet

    %   special case for Sparse Structure-Regularized Learning with Least
    %   Squares Loss when rho2 = 0, R = I-1/T*ones;

    addpath('../../MALSAR/functions/Tree_based/');
    addpath('../train_and_test/');
    % addpath('../../MALSAR/utils/');

    % load data
    X = data.X;
    Y = data.Y;

    all_trial = 5;
    all_rmse = zeros(3, all_trial);
    % all_perf = zeros(8, all_trial);

    for tt = 1:all_trial
        % split data into training and testing.
        training_percent = 0.8;
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
        % param_range = [0.1 10 1000 100000];
        % alpha_range = [0.25 0.5 0.75 1];
        % param_range = [0.05 0.1 10 50 100];
        % alpha_range = [0.9 0.95 1.0];
        p1 = [100, 200, 400];
        p2 = [0.1, 0.2, 0.4];
        p3 = [0.05 0.1 0.2];
        p4 = 0.9;

        ff = @(x, y, rho1, rho2, rho3, alpha, opts) ...
            SolveTreeBased_ElasticNet(x, y, Clusters, rho1, rho2, rho3, alpha);

        fprintf('Perform model selection via cross validation: \n')
        % [ best_param, ~ ] = CrossValidation4Param_alpha...
        %     ( X_tr, Y_tr, ff, opts, param_range, param_range, param_range, ...
        %     alpha_range, cv_fold, eval_func_str, higher_better);
        [ best_param, ~ ] = CrossValidation4Param_alpha...
            ( X_tr, Y_tr, ff, opts, p1, p2, p3, p4, ...
            cv_fold, eval_func_str, higher_better);

        % build model using the optimal parameter
        fixedW = SolveTreeBased_ElasticNet(X_tr, Y_tr, Clusters, ...
            best_param(1), best_param(2), best_param(3), best_param(4));

        % show final performance
        [f_mse, f_rss, f_tss] = eval_MTL_mse(Y_te, X_te, fixedW);

        all_rmse(:, tt) = [f_mse, f_rss, f_tss];
    end

    fixedErrors = zeros(all_trial, 4);
    fixedErrors(:,1:3) = all_rmse';
    fixedErrors(:,4) = 1 - ( fixedErrors(:, 2) ./ fixedErrors(:, 3) );

    save('fixedSchoolW.mat','fixedW');
    save('fixedSchoolErrors.mat','fixedErrors');
    save('fixedSchoolBest.mat','best_param');
    % save('perf.mat','perform_mat');

