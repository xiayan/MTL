%% SCRIPT test_script.m
%   Multi-task learning training/testing example. This example illustrates
%   how to perform split data into training part and testing part, and how
%   to use training data to build prediction model (via cross validation).
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
%
%% Related functions
%   mtSplitPerc, CrossValidation1Param, Least_Trace

clear; clc;

addpath('../../MALSAR/functions/msmtfl/');
addpath('../../MALSAR/utils/');


% load data
load_data = load('../../data/school.mat');

X = load_data.X;
Y = load_data.Y;

% preprocessing data
for t = 1: length(X)
    X{t} = zscore(X{t});                  % normalization
    X{t} = [X{t}(:,1:end-1) ones(size(X{t}, 1), 1)]; % add bias.
end

all_trial = 1;
% model parameter range
pr1 = [0.1 1 10 100];
pr2 = [1 10 100 1000];

all_rmse = zeros(3, all_trial);
% all_perf = zeros(length(param_range), all_trial);

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


fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation2Param...
    ( X_tr, Y_tr, 'Least_msmtfl_capL1', opts, pr1, pr2, cv_fold, eval_func_str, higher_better);

% disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter
W = Least_msmtfl_capL1(X_tr, Y_tr, best_param(1), best_param(2), opts);

% show final performance
[f_mse, f_rss, f_tss] = eval_MTL_mse(Y_te, X_te, W);
% fprintf('Performance on test data: %.4f\n', final_performance);

all_rmse(:, tt) = [f_mse; f_rss; f_tss];
% all_perf(:, tt) = perform_mat;

end

