clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
% [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������
RGB=imread('altlas11.jpg');
%  RGB=im2double(RGB);%%��Ӱ��
tic
R= RGB(:, :, 1); 
G= RGB(:, :, 2); 
B= RGB(:, :, 3); 

HSV=rgb2hsv(RGB);%ת��HSV
 toc                                                              % % figure
%                                                                             % % imshow(YUV);
% clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
%  clc
% clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
%  clc
%  [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
%  
%  RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', 'ԭͼ');%%figure������
% tic
% % RGB=imread('timg.jpg');
% R= RGB(:, :, 1); 
% G= RGB(:, :, 2); 
% B= RGB(:, :, 3); 
% 
% HSV=rgb2hsv(RGB);%ת��HSV
%     toc
%     
