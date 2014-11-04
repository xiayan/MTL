function kde = kde_train(X,s,n)
  [W, kde.B, kde.G, kde.P, kde.S] = fastfood(X,s,n);
  kde.K = mean(exp(-1i*W),2);
  kde.s = s;
  kde.n = n;
  kde.c = 1/(2*pi)^(size(X,1)/2)*det(diag(s))/n;
