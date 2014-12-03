function [] = plotHM(titleStr, data, range)

    fig=figure(2);
    imagesc(data);
    colormap('bone');
    xlabel('Tasks', 'FontSize', 14);
    ylabel('Features', 'Fontsize',14);
    title(titleStr,  'FontSize', 15);
    colorbar;
    caxis([-range,range]);
    set(fig,'position',[10,10,500,350]);
end