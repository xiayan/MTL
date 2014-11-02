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
%   mtSplitPerc, CrossValidation1Param_withR, Least_Trace

%construct graph R
clear;
clc;
close;

addpath('../../MALSAR/functions/Lasso/'); % load function
addpath('../../SLEP_4.0/SLEP/functions/invCov/'); %load sparse inverse covariance from SLEP
addpath('../../SLEP_4.0/SLEP/cFiles/spInvCoVa/');
addpath('../../MALSAR/functions/SRMTL/'); % load function
addpath('../../MALSAR/utils/'); % load utilities

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 500; % maximum iteration number of optimization.

load('../../data/school.mat'); % load sample data.
task_num = length(X);

% use Lasso calculate a model (used for graph analysis)
[W_pre] = Least_Lasso(X, Y, 0.01, opts);

% normalize matrix.
mean_1=mean(W_pre,1);
W_pre=W_pre-repmat(mean_1,size(W_pre, 1),1);
norm_2=sqrt( sum(W_pre.^2,1) );
W_pre=W_pre./repmat(norm_2,size(W_pre, 1),1);



% use sparse inverse covariance to calculate a graph
S=W_pre'*W_pre; % empirical covariance matrix 
sinv_opts.maxIter=100;
sinv_opts.lambda=0.1;
Theta=sparseInverseCovariance(S, sinv_opts.lambda, sinv_opts);

graph = Theta~=0;
graph = graph - eye(task_num);
edge_num = nnz(graph)/2;
fprintf('%u edges are found\n', edge_num);

imshow(1- graph, 'InitialMagnification', 'fit')
title(sprintf('Sparse Inverse Covariance Graph (lambda=%.2f, #edge = %u)', sinv_opts.lambda, edge_num));
print('-dpdf', '-r300', 'LeastSRMTLExp_spinv_1');

% construct graph structure variable.
R = [];
for i = 1: task_num
    for j = i + 1: task_num
        if graph (i, j) ~=0
            edge = zeros(task_num, 1);
            edge(i) = 1;
            edge(j) = -1;
            R = cat(2, R, edge);
        end
    end
end

%%


addpath('../../MALSAR/functions/joint_feature_learning/');
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

all_trial = 10;
all_rmse = zeros(3, all_trial);
all_perf = zeros(8, all_trial);

parfor tt = 1:all_trial

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
param_range = [0.001 0.01 0.1 1 10 100 1000 10000];

ff=@(x,y,rho1,rho2,opts)Least_SRMTL(x,y,R,rho1,rho2,opts);

fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation2Param_withR...
    ( X_tr, Y_tr, ff, opts, param_range,param_range, cv_fold, eval_func_str, higher_better);

% disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter
W = Least_SRMTL(X_tr, Y_tr, R, best_param(1), best_param(2), opts);

% show final performance
[f_mse, f_rss, f_tss] = eval_MTL_mse(Y_te, X_te, W);
% fprintf('Performance on test data: %.4f\n', final_performance);

all_rmse(:, tt) = [f_mse, f_rss, f_tss];
% all_perf(:, tt) = perform_mat;

end
%%
save('tmp_graph_sp_invcov.mat')