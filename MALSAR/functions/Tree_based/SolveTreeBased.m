function [ W,Theta ] = SolveTreeBased( X,Y, cluster,lambda1, lambda2, lambda3 )
%SOLVETREEBASED Summary 

% clusters is a column vector indicating which task is assigned to which cluster
% center theta i. i = 1,2,...,t
% please produce the tree structure into this cluster format!

% number of clusters
t=max(cluster);
% number of tasks
T=length(X);
%feature dimention, all tasks have the same features
d=size(X{1},2);
% 
% nlist=zeros(T,1);
% for i=1:T
%     nlist(i)=size(X{i},2);
% end

XTX=cell(T,1);
b=zeros(T*d+t*d,1);
cur=0;

%construct the sparse matrix
idxi=[];
idxj=[];
s=[];

%construct the diagonal blocks for the w part
for i=1:T
    XTX{i} = X{i}'*X{i}+(lambda1+lambda2)*eye(d);
    [ii,jj,ss]=find(XTX{i});
    idxi=[idxi;ii+cur];
    idxj=[idxj;jj+cur];
    s=[s;ss];    
    b(cur+1:cur+d)=X{i}'*Y{i};
    cur=cur+d;
end

% diagonal block for hte theta part
[ii,jj,ss]=find((lambda2+lambda3)*speye(t*d));
for i=1:t
    sz=sum(cluster==i);
    ss(((i-1)*d+1):i*d)=ss(((i-1)*d+1):i*d)*sz;
end
    idxi=[idxi;ii+d*T];
    idxj=[idxj;jj+d*T];
    s=[s;ss];
    
% Off diagonal block;
for i=1:t
    %make sure cluster is a column vector!
    %otherwise this breaks!
    A=kron(cluster==i,-lambda2*speye(d));
    [ii,jj,ss]=find(A);
    idxi=[idxi;ii];
    idxj=[idxj;jj+d*T+(i-1)*d];
    s=[s;ss];
    
    [ii,jj,ss]=find(A');
    idxi=[idxi;ii+d*T+(i-1)*d];
    idxj=[idxj;jj];
    s=[s;ss];
end
    
    
AA = sparse(idxi,idxj,s,d*T+t*d,d*T+t*d);

%solve it
w=AA\b; % perhaps use PCG instead if there are many tasks

W=reshape(w(1:d*T),[d,T]);
Theta=reshape(w((d*T+1):end),[d,t]);

end

