function [best_param, perform_mat] = find_best_params(func_obj_str, num_params)
% find the best parameter using cross validation
% for methods contain 1, 2 or 3 parameters

if num_params ~= 1 && num_params ~= 2 && num_params ~= 3
    error('\n Cannot use this function. See the first line of comment \n');
end

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
[X_tr, Y_tr, ~, ~] = mtSplitPerc(X, Y, training_percent);

% the function used for evaluation.
eval_func_str = 'eval_MTL_mse';
higher_better = false;  % mse is lower the better.

% cross validation fold
cv_fold = 5;

% optimization options
opts = [];
opts.maxIter = 100;

% model parameter range
param1_range = [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000 5000 10000];
param2_range = [0.001 0.01 0.1 1 10 100 1000 10000];
param3_range = [0.001 0.01 0.1 1 10 100 1000 10000];
% param1_range = [1 10 100];
% param2_range = [1 10 100];
% param3_range = [1 10 100];

fprintf('Perform model selection via cross validation: \n')
if num_params == 1
    [ best_param, perform_mat ] = CrossValidation1Param...
    ( X_tr, Y_tr, func_obj_str, opts, param1_range, cv_fold, eval_func_str, higher_better);
elseif num_params == 2
    [ best_param, perform_mat] = CrossValidation2Param...
    ( X_tr, Y_tr, func_obj_str, opts, param1_range, param2_range, cv_fold, eval_func_str, higher_better);
elseif num_params == 3
    [ best_param, perform_mat] = CrossValidation3Param...
    ( X_tr, Y_tr, func_obj_str, opts, param1_range, param2_range, param3_range, cv_fold, eval_func_str, higher_better);

end

