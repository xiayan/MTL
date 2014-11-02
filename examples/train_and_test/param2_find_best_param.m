%% SCRIPT test_script.m
function [best_param, perform_mat] = param2_find_best_param(func_obj_str)

addpath(genpath('../../MALSAR/'));
addpath('../../MALSAR/utils/');

% load data
load_data = load('../../data/school.mat');

X = load_data.X;
Y = load_data.Y;

% preprocessing data
for t = 1: length(X)
	tmp = zscore(X{t}(:, 1:end-1));  % normalize except the bias col
	X{t} = [tmp ones(size(X{t}, 1), 1)]; % add bias.
end

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
param1_range = [0.001 0.01 0.1 1 10 100 1000 10000];
param2_range = [0.001 0.01 0.1 1 10 100 1000 10000];

fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation2Param...
    ( X_tr, Y_tr, func_obj_str, opts, param1_range, param2_range, cv_fold, eval_func_str, higher_better);

end

