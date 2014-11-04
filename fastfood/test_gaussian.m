n = 4000;     % n: number of samples
S = eye(2^2); % S: (co)variance of the data
k = 2^11;     % k: number of random features
r = 1;        % r: number of experiment repetitions

%function res = test_gaussian(n,S,k,r)

d   = size(S,1);
res = [];

for i=1:r
  x = S*randn(d,n);
  y = S*randn(d,n);
  
  kde_full = kde(x, 'rot');
  
  s = 1./getBW(kde_full,1);
  
  kde_fast = kde_train(x,s,k);
  
  tic;
  kde_full_lik = evaluate(kde_full,y)';
  kde_full_time = toc;
  
  tic;
  kde_fast_lik = kde_test(kde_fast,y);
  kde_fast_time = toc;

  kde_fast_err = mean(abs(kde_full_lik-kde_fast_lik));

  res = [res; [kde_full_time kde_fast_time kde_fast_err]];

  plot(kde_full_lik,kde_fast_lik,'.',kde_full_lik,kde_full_lik,'.')
  title(['d='               num2str(d)...
         ', n='             num2str(n)...
         ', k='             num2str(k)...
         ', KDE-time='      num2str(kde_full_time)...
         ', FastFood-time=' num2str(kde_fast_time)]);
end;

res
