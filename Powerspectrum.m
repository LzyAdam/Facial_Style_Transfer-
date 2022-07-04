clear%%千万不能没有要不然，有些莫名其妙的错
 clc
% [R,G,B]=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令
RGB=imread('altlas11.jpg');
%  RGB=im2double(RGB);%%不影响
tic
R= RGB(:, :, 1); 
G= RGB(:, :, 2); 
B= RGB(:, :, 3); 

HSV=rgb2hsv(RGB);%转成HSV
 toc                                                              % % figure
%                                                                             % % imshow(YUV);
% clear%%千万不能没有要不然，有些莫名其妙的错
%  clc
% clear%%千万不能没有要不然，有些莫名其妙的错
%  clc
%  [fn,pn,fi]=uigetfile('*.jpg','选择图片');
%  
%  RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字
% tic
% % RGB=imread('timg.jpg');
% R= RGB(:, :, 1); 
% G= RGB(:, :, 2); 
% B= RGB(:, :, 3); 
% 
% HSV=rgb2hsv(RGB);%转成HSV
%     toc
%     
