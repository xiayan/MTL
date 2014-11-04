function [W,B,G,P,S] = fastfood(X,s,n,B,G,P,S)
  X = [X; zeros(2^nextpow2(size(X,1))-size(X,1), size(X,2))];
  n = 2^nextpow2(n);
  d = size(X,1);
  W = zeros(n,size(X,2));

  if nargin < 7
    [B,G,P,S] = deal(zeros(n/d,d));

    for i=1:(n/d)
      idx = (d*(i-1)+1):(d*i);
      [W(idx,:),B(i,:),G(i,:),P(i,:),S(i,:)] = fastfood_block(X,s);
    end;
  else
    for i=1:(n/d)
      idx = (d*(i-1)+1):(d*i);
      W(idx,:) = fastfood_block(X,s,B(i,:),G(i,:),P(i,:),S(i,:));
    end;
  end;
