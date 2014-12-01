function [ W,Theta,fval ] = SolveTreeBased_ElasticNet( X,Y, cluster,lambda, lambda2, lambda3, alpha)
% Mostly the same as previous one except lambda1.
% lambda2 is for the 0.5||W_i-Theta_Parent||^2
% lambda3 is for 0.5||Theta||_F^2
% [lambda,alpha] is for the elastic net penalty
%  lambda(0.5*(1-alpha) ||W||_F^2 + alpha||W||_1),  


% so please tune \alpha by choosing alpha between 0 and 1
% 1 is pure lasso, 0 is pure ridge.

lambda1 = (1-alpha)*lambda;
lambda4 = alpha*lambda;

% get dimensions
    % number of clusters 
    t=max(cluster);
    % number of tasks
    T=length(X);
    %feature dimention, all tasks have the same features
    d=size(X{1},2);

%     L=FindLipschitz( X,Y, cluster,lambda1, lambda2, lambda3 );
%     %strongly covnex parameter
%     mu=min(lambda2,lambda4);
%     opts.MU=mu;
%     opts.STEP_SIZE = 1/L;
%     opts.FIXED_STEP_SIZE=true;
    
% Solve using APG
opts.MAX_ITERS=1000;
opts.USE_RESTART=true;

opts.QUIET = true; % if false writes out information every 100 iters
opts.GEN_PLOTS = false; % if true generates plots of norm of proximal gradient


grad_f=@(w,opts) compute_grad(w,X,Y,cluster,lambda1, lambda2, lambda3);
prox_h=@(x,t,opts) soft_thresh(x,t);
w=apg(grad_f,prox_h,d*T+d*t,opts);


%output
W=reshape(w(1:d*T),[d,T]);
Theta=reshape(w((d*T+1):end),[d,t]);
fval=compute_obj_fun( w,X,Y,cluster,lambda1, lambda2, lambda3,lambda4);


end


% prox operator
function y=soft_thresh(x,lambda)
    y=max(x-lambda,0) + min(x+lambda,0);
end

% gradient computation
function grad=compute_grad( w,X,Y,cluster,lambda1, lambda2, lambda3)
    % number of clusters 
    t=max(cluster);
    % number of tasks
    T=length(X);
    %feature dimention, all tasks have the same features
    d=size(X{1},2);

    
    grad=zeros(size(w));
    
    
    W=reshape(w(1:d*T),[d,T]);
    Theta=reshape(w((d*T+1):end),[d,t]);
    for i=1:T
        grad(((i-1)*d+1):(i*d))=X{i}'*(X{i}*W(:,i)-Y{i})...
            +(lambda1+lambda2)*W(:,i)-lambda2*Theta(:,cluster(i));
    end
    for i=1:t
        grad(d*T+(i-1)*d+1:d*T+i*d) = -lambda2*sum(W(:,cluster==i),2)...
            + (lambda2+lambda3)*Theta(:,i);
    end
    
end


% gradient computation
function obj=compute_obj_fun( w,X,Y,cluster,lambda1, lambda2, lambda3,lambda4)
    % number of clusters 
    t=max(cluster);
    % number of tasks
    T=length(X);
    %feature dimention, all tasks have the same features
    d=size(X{1},2);

    W=reshape(w(1:d*T),[d,T]);
    Theta=reshape(w((d*T+1):end),[d,t]);
    obj=0;
    for i=1:T
        obj=obj+0.5*norm(X{i}*W(:,i)-Y{i})^2;        
    end
    for i=1:t
        obj=obj+0.5*lambda2*norm(bsxfun(@minus,W(:,cluster==i),Theta(:,i)),'fro')^2;
    end
    obj=obj+0.5*lambda1*norm(W,'fro')^2 + 0.5*lambda3*norm(Theta,'fro');
    obj=obj+lambda4*norm(W(:),1);
    
end




function [ L ] = FindLipschitz( X,Y, cluster,lambda1, lambda2, lambda3 )
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

L=svds(AA,1);

end