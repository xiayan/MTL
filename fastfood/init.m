function [] = init()
	M = csvread('init-split');
	[W,B,G,P,S] = fastfood(M(3:3246)',1,4096); 
	save('init.mat', 'B','G','P','S');
end
