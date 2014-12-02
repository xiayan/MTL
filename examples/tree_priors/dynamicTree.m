%% main function
function [final_W, final_p, s] = dynamicTree(data, Iterations)
    % Input:
    % data is a cell array name. It should follow the same convention
    % as the school dataset
    % Iteration is the number of total times to perform tree proposals
    %
    % Output:
    % final_W: final set of weights. feature * tasks
    % final_p: final test mse
    % s: final tree structure

    num_tasks = length(data.X);
    % super-class vector. Two tasks are similar if they have the same number in
    % the s vector
    % initialize tree to all have the same super class
    s = ones(num_tasks, 1);

    % generate a split of data here. 'TH' means 'Train' plus 'Hold out' data
    [THX, THY, testX, testY] = splitDataset(data, 0.8);
    TH.X = THX;
    TH.Y = THY;

    for i = 1:Iterations
        order = randperm(num_tasks); % generate a random order for tree proposal
        last_s = s;

        for t = 1:num_tasks
            % for each task, propose S+1 tree proposals.
            % Train each tree proposal using a same train set
            % Test proposals on a same test set, and keep the best tree
            cur_task = order(t);

            % S contains all different trees. tasks * proposal
            S = treeProposals(s, cur_task);
            [trainX, trainY, holdX, holdY] = splitDataset(TH, 0.8);

            % cur_W: feature * task * proposal
            cur_W = trainTrees(S, trainX, trainY);
            % cur_P: proposal * 1
            cur_P = testTrees(cur_W, holdX, holdY);
            min_idx = min(cur_P);

            s = S(:, min_idx);
        end

        % terminate if the s vector does not change
        if isequal(last_s, s)
            break
        end
    end

    % train the whole data again using the converged s vector.
    final_W = trainTrees(s, THX, THY);
    final_p = testTrees(final_W, testX, testY);

end


%% splitDataset
function [X1, Y1, X2, Y2] = splitDataset(data, percent)
    addpath('../train_and_test/');
    [X1, Y1, X2, Y2] = mtSplitPerc(data.X, data.Y, percent);

end


%% treeProposals
function S = treeProposals(s, cur_task)
    % determine the total number of parents
    parents = sort( unique(s) );
    num_tasks = length(s);
    num_pps = length(parents) + 1;
    S = zeros(num_tasks, num_pps);

    % assign the cur_task to each of existing parents
    for i = 1:length(parents)
        cur_s = s;
        cur_s(cur_task) = parents(i);
        S(:, i) = cur_s;
    end

    % assign the cur_task to a new parent
    new_s = s;
    new_s(cur_task) = parents(end) + 1;
    S(:, end) = new_s;

end


%% trainTrees
function W = trainTrees(S, X, Y)
    % add the path to optimization functions
    addpath('../../MALSAR/functions/Tree_based/');

    % W: feature * task * proposal
    num_feats = size(X{1}, 2);
    num_tasks = length(X);
    num_pps   = size(S, 2);
    W = zeros(num_feats, num_tasks, num_pps);

    for i = 1:num_pps
        s = S(:, i);
        % TODO: should I change the weight of current task to the
        % theta_parent here?
        cur_W = Su_Optimization(X, Y, s);
        W(:, :, i) = cur_W;
    end

end


%% Su_Optimization
function W = Su_Optimization(X, Y, s)
    addpath('../train_and_test/');

    eval_func_str = 'eval_MTL_mse';
    higher_better = false;  % mse is lower the better.

    % cross validation fold
    cv_fold = 5;

    % optimization options
    opts = [];
    opts.maxIter = 100;

    % model parameter range
    param_range = [0.001 0.01 0.1 1 10 100 1000 10000];
    alpha_range = [0 0.25 0.5 0.75 1];

    ff = @(x, y, rho1, rho2, rho3, alpha, opts) ...
        SolveTreeBased_ElasticNet(x, y, s, rho1, rho2, rho3, alpha);

    best_param = CrossValidation4Param_alpha...
        ( X, Y, ff, opts, param_range, param_range, param_range, ...
        alpha_range, cv_fold, eval_func_str, higher_better );

    % build model using the optimal parameter
    W = SolveTreeBased_ElasticNet( X, Y, s, best_param(1), best_param(2), ...
        best_param(3), best_param(4) );
end


%% testTrees
function perf = testTrees(W, X, Y)
    % perf: proposal * 1
    num_pps = size(W, 3); % number of proposals
    perf = zeros(num_pps, 1);

    for i = 1:num_pps
        cur_W   = W(:, :, i);
        cur_mse = evalW(X, Y, cur_W);
        perf(i) = cur_mse;
    end

end


%% evalW
function [mse, rss, tss] = evalW(X, Y, W)
    % evaluate a set of weight W. Copied from 'eval_MTL_mse.m'
    task_num = length(X);
    mse = 0; % mean squared error
    rss = 0; % total residual
    tss = 0; % mean total squared error
    % Calculate as R^2 = 1 - mse/mts

    total_sample = 0;
    for t = 1: task_num
        y_pred = X{t} * W(:, t);
        mse = mse + sqrt(sum((y_pred - Y{t}).^2)) * length(y_pred);

        rss = rss + sum((y_pred - Y{t}).^2);
        tss = tss + sum((mean(Y{t}) - Y{t}).^2);
        total_sample = total_sample + length(y_pred);
    end
    mse = mse./total_sample;
end


% %% crossValidation
% function [cur_performance] = crossValidation(TH, S, fold)
%     % split the 'th' set into train and hold-out set
%     cur_performance = zeros( size(S, 2), fold );
%
%     for f = 1:fold
%         % don't use splitDataset. Use the cross validation code
%         [trainX, trainY, holdOutX, holdOutY] = splitDataset(TH, 0.8);
%
%         % train all tree proposals
%         % cur_W: proposal * task * features
%         % cur_theta: proposal * task * 1
%         % use Su's code as trainTrees
%         [cur_W, cur_Theta] = trainTrees(S, trainX, trainY);
%         % test all proposed tree weights on the hold out set
%         % cur_P: proposal * 1
%         cur_P = testTrees(cur_W, cur_Theta, holdOutX, holdOutY);
%         cur_performance(:, f) = cur_P;
%
%     end
%     cur_performance = mean(cur_performance, 2);
%
% end

