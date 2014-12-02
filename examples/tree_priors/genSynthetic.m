function [data, W, cluster_index] = genSynthetic(clus_num, clus_task_num)
    clus_var = 90;  % cluster variance
    task_var = 2;   % inter task variance
    nois_var = 5;   % variance of noise

    % clus_num = 3;                        % clusters
    % clus_task_num = 3;                   % task number of each cluster
    task_num = clus_num * clus_task_num;   % total task number.
    dimension   = 20;        % total dimension
    comm_dim    = 2;         % independent dimension for all tasks.
    clus_dim    = 3;         % dimension of cluster
    dimension   = dimension - comm_dim;

    % generate cluster model
    cluster_weight = randn(dimension, clus_num) * clus_var;
    for i = 1: clus_num
        cluster_weight (randperm(dimension) > clus_dim, i) = 0;
    end
    W = repmat (cluster_weight, 1, clus_task_num);
    cluster_index = repmat (1:clus_num, 1, clus_task_num)';

    % generate task and intra-cluster variance
    W_it = randn(dimension, task_num) * task_var;
    for i = 1: task_num
        W_it(W(:, i) == 0, i) = 0;
    end
    W = W + W_it;

    % add in common weight for all tasks
    common = repmat(randn(comm_dim, 1), 1, task_num) * clus_var;
    noise  = randn(comm_dim, task_num) * nois_var;
    common = common + noise;

    W = [W; common];

    dimension = dimension + comm_dim;

    % Generate Input/Output
    X = cell(task_num, 1);
    Y = cell(task_num, 1);

    % sample size for each task is uniformly distributed from 15 - 25
    a = dimension - 5;
    b = dimension + 5;
    for i = 1: task_num
        sample_size = floor( (b-a) .* rand(1, 1) + a );
        X{i} = randn(sample_size, dimension);
        xw   = X{i} * W(:, i);
        noise = randn(size(xw)) * nois_var;
        Y{i} = xw + noise;
    end

    data.X = X;
    data.Y = Y;

