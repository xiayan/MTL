function p = kde_test(kde,X)
  FF = exp(-1i*fastfood(X,kde.s,kde.n,kde.B,kde.G,kde.P,kde.S));
  p  = real(kde.c*FF'*kde.K);
