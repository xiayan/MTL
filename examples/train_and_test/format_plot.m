%% script to plot regularization path

fig=figure(1)
 semilogx(param_range,perform_mat(:,9),'-x','linewidth',2,'markersize',14);
 xlabel('\lambda_1','fontsize',14);
 ylabel('MSE','fontsize',14);
 grid on
 set(fig,'position',[10,10,500,350]);
 
 %%
 fig=figure(2)
imagesc(W(1:28,:))
colormap('bone')
xlabel('Tasks', 'FontSize', 14)
ylabel('Features', 'Fontsize',14)
colorbar
  set(fig,'position',[10,10,500,350]);