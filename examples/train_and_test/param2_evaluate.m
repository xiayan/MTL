function [mse_s, tss_s, rss_s, rsq_s, W] = param2_evaluate(func_obj_str, params, repetition)

% W is the weight matrix it learned at the last

num_metrics = 4;

addpath(genpath('../../MALSAR/'));
addpath('../../MALSAR/utils/');


% load data
load_data = load('../../data/school.mat');

X = load_data.X;
Y = load_data.Y;

% preprocessing data
for t = 1: length(X)
    X{t} = zscore(X{t});                  % normalization
    X{t} = [X{t} ones(size(X{t}, 1), 1)]; % add bias. 
end

% split data into training and testing.
training_percent = 0.8;

% optimization options
opts = [];
opts.maxIter = 100;

% the function used for evaluation.
eval_func_str = 'eval_MTL_mse';


mse_s = zeros(repetition, 1);
tss_s = zeros(repetition, 1);
rss_s = zeros(repetition, 1);
rsq_s = zeros(repetition, 1);

for rep=1:repetition
	[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);

	func = str2func(func_obj_str);
	W = func(X_tr, Y_tr, params(1), params(2), opts);
	[mse, rss, tss] = eval_MTL_mse(Y_te, X_te, W);
	mse_s(rep, 1) = mse;
	rss_s(rep, 1) = rss;
	tss_s(rep, 1) = tss;
	rsq_s(rep, 1) = 1 - rss / tss;
end

