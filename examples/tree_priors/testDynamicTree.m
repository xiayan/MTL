function [all_W, dynamicErrors, all_s] = testDynamicTree(data)
    %%
    % data should follow the school dataset format. It should also be renormalized
    %
    %% Related functions
    %   mtSplitPerc, CrossValidation4Param_alpha, SolveTreeBased_ElasticNet

    addpath('../../MALSAR/functions/Tree_based/');
    addpath('../train_and_test/');

    num_feats = size(data.X{1}, 2);
    num_tasks = length(data.X);

    all_trial = 1;
    all_rmse  = zeros(3, all_trial);
    all_s     = zeros(num_tasks, all_trial);
    all_W     = zeros(num_feats, num_tasks, all_trial);

    parpool('local');

    for tt = 1:all_trial
        % split data into training and testing.
        [trainX, trainY, testX, testY] = mtSplitPerc(data.X, data.Y, 0.8);
        train.X = trainX;
        train.Y = trainY;
        [W, s] = dynamicTree(train, 100);

        % show final performance
        [f_mse, f_rss, f_tss] = eval_MTL_mse(testY, testX, W);

        all_rmse(:, tt) = [f_mse, f_rss, f_tss];
        all_s(:, tt)    = s;
        all_W(:, :, tt) = W;
    end

    dynamicErrors = zeros(all_trial, 4);
    dynamicErrors(:,1:3) = all_rmse';
    dynamicErrors(:,4) = 1 - ( dynamicErrors(:, 2) ./ dynamicErrors(:, 3) );

    delete(gcp);

    save('dynamicW.mat','all_W');
    save('dynamicErrors.mat','dynamicErrors');
    save('dynamicS.mat','all_s');

