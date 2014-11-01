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

addpath('../../MALSAR/functions/low_rank/');
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
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);

param_range = [0.001 0.01 0.1 1 10 100 1000 10000];
num_repeat = 10;

% train a single model using all samples
t_X = cell2mat(X_tr');
t_y = cell2mat(Y_tr');
[B, FitInfo] = lasso(t_X, t_y, 'Alpha', 1, 'CV', 5, 'Lambda', param_range, ...
'Standardize', false);
% [B, FitInfo] = lasso(t_X, t_y, 'Alpha', 1, 'CV', 5);
[~, idx] = min(FitInfo.MSE);
w = B(:,idx);

num_tasks = length(X_tr);
W = repmat(w, 1, num_tasks);

[mse, tts] = eval_MTL_mse(Y_te, X_te, W);
fprintf('Performance on test data: %.4f, %.4f\n', mse, tts);

