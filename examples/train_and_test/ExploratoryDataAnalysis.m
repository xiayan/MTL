%% plotting the clustering of schools


addpath('../../MALSAR/functions/joint_feature_learning/');
addpath('../../MALSAR/utils/');
addpath('../../fastfood');

% load data
load_data = load('../../data/school.mat');

X = load_data.X;
Y = load_data.Y;

%%k-means clustering


%% plot y for each shool against school idx
p=size(X{1},2); 
pp= 2^10;%1024 fastfood features
Ymean = zeros(length(Y),1);
Yvar=Ymean;
Xmean=zeros(p,length(Y));
X_ff=zeros(pp,length(Y));
%construct fastfood transform
[W,B,G,P,S] = fastfood(randn(p,1),1,pp); 

%%
for i=1:length(Y)
    Ymean(i)=mean(Y{i});
    Yvar(i)=std(Y{i});
    
    Xmean(:,i)=mean(X{i},1);
    %computer mean embedding
    X_ff(:,i) = mean(fastfood(X{i}',1,pp,B,G,P,S),2); 
end

%%
[ymeansorted, idx]=sort(Ymean);
figure(1)
errorbar(1:139,ymeansorted,Yvar(idx));
% hold off
% plot(ymeansorted,'b*');
% hold on
% plot(ymeansorted+Yvar(idx),'g.')
% plot(ymeansorted-Yvar(idx),'g.')

%% clustering of schools by their scores
k=3;
[Class, centroid]=kmeans(Ymean,k);


schoolidx=1:139;
figure(2)
hold all
for i=1:k
plot(schoolidx(Class==i),Ymean(Class==i),'*');
end

%% clustering the mean and visualization
k=3;

[Class, centroid]=kmeans(Xmean',k);

[U, Sigma, V]=svd(Xmean);
Xmean_proj=Sigma(1:2,1:2)*V(:,1:2)';
figure(3)
hold all
for i=1:k
plot(Xmean_proj(1,Class==i),Xmean_proj(2,Class==i),'x','markersize',14,'linewidth',2);
end




%% clustering the mean embedding and visualization

k=3;

[Class, centroid]=kmeans(X_ff,k);

%
[U, Sigma, V]=svd(X_ff');
%
X_ff_proj=Sigma(1:2,1:2)*V(:,1:2)';
%
figure(3)
hold all
for i=1:k
plot(X_ff_proj(1,Class==i)',X_ff_proj(2,Class==i)','x','markersize',14,'linewidth',2);
end


