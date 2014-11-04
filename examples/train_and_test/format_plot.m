%% script to plot regularization path

fig=figure(1)
 semilogx(param_range,perform_mat(:,1),'-x','linewidth',2,'markersize',14);
 xlabel('\lambda','fontsize',20);
 ylabel('MSE','fontsize',14);
 grid on
 set(fig,'position',[10,10,500,400]);
 
 %%
 fig=figure(2)
imagesc(W(1:27,:))
colormap('bone')
xlabel('Tasks', 'FontSize', 14)
ylabel('Features', 'Fontsize',14)
colorbar
  set(fig,'position',[10,10,500,400]);