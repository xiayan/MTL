%compile_spiral;                % compile spiral mex routine
X = randn(2^10,1);             % generate a 1024-dim vector
s = 1;                         % scale equal to one
n = 2^12;                      % we want 2048 features
[W,B,G,P,S] = fastfood(X,s,n); % generate new n-dim fastfood for X with scale s
W2 = fastfood(X,s,n,B,G,P,S);  % compute previous fastfood(B,G,P,S) again on same vector
all(W==W2)                     % matches?
