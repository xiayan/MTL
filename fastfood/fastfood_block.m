function [W,B,G,P,S] = fastfood_block(X,s,B,G,P,S)
  d = size(X,1);                             % d is assumed power of 2
  if nargin < 6                              % 2 arguments: generate pars
    B = 2*unidrnd(2,1,d)-3;                  % bernoulli diagonal
    G = randn(1,d);                          % gaussian diagonal
    P = randperm(d);                         % permutation indices
    S = s;                                   % scaling diagonal (TODO!!!)
  end;                                       % 6 arguments: use pars
  %W = spiral_fwht(sparse(1:d,1:d,B)*X);      % W = hadamard(BX)
  W = hadamardc(sparse(1:d,1:d,B)*X);
  %W = spiral_fwht(sparse(1:d,1:d,G)*W(P,:)); % W = hadamard(GPW))
  W = hadamardc(sparse(1:d,1:d,G)*W(P,:));
  W = 1/sqrt(d).*sparse(1:d,1:d,S)*W;
  
