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

addpath('../../MALSAR/functions/Lasso/');
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

% split data into training and testing.
training_percent = 0.8;
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);

param_range = [0.001 0.01 0.1 1 10 100 1000 10000];
% cross validation fold
cv_fold = 5;
% optimization options
opts = [];
opts.maxIter = 100;

num_repeat = 10;

% train a single model using all samples
% t_X = cell2mat(X_tr');
% t_y = cell2mat(Y_tr');
% [B, FitInfo] = lasso(t_X, t_y, 'Alpha', 1, 'CV', 5, 'Lambda', param_range, ...
% 'Standardize', false);
% lassoPlot(B,FitInfo,'PlotType','CV');
% % [B, FitInfo] = lasso(t_X, t_y, 'Alpha', 1, 'CV', 5);
% [~, idx] = min(FitInfo.MSE);
% w = B(:,idx);

% num_tasks = length(X_tr);
% W = repmat(w, 1, num_tasks);

% train a single model using package Lasso
all_X_tr = {cat(1, X_tr{:})};
all_y_tr = {cat(1, Y_tr{:})};
[ best_param, perform_mat] = CrossValidation1Param...
( all_X_tr, all_y_tr, 'Least_Lasso', opts, param_range, cv_fold, 'eval_MTL_mse', false);
w = Least_Lasso(all_X_tr, all_y_tr, best_param, opts);

all_X_te = {cat(1, X_te{:})};
all_y_te = {cat(1, Y_te{:})};
[mse, rss, tss] = eval_MTL_mse(all_y_te, all_X_te, w);

% The TSS here is calculated different than other MTL methods.
% Use the MTL version of TSS to calculate R^2.
fprintf('Performance on test data: %.4f, %.4f, %.4f\n', mse, rss, tss);

