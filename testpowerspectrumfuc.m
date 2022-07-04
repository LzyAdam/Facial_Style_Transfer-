
% % [fn,pn,fi]=uigetfile({'*.bmp' ;'*.jpg'},'选择图片');
% % 
% % I=imshow([pn fn ]);
%subplot(221);
%b=fft2(I);
%c=imshow(b);
% I=imread([pn fn fi]);
% I=rgb2yuv(I);
% I=rgb2gray(I);
% %=im2double(I);
% F=fft2(I);
% F=fftshift(F);
% F=abs(F);
% T=log(F+1);
% figure;
% 
% % % A=imshow(T,[]);
% % %  B=sumsqr(F);
% % % disp(B);
% %  A=[0,1,2,3;4,5,6,7]
% %  
% %  sumsqr(A)
% % 
% %  

% % [m n dim]=uigetfile({'*.bmp' ;'*.jpg'},'选择图片');
% % % % 
% % I=imshow([m n dim ]);
% % img=im2double(I);
% % % % img=imread('lena_color.jpg');
% % img=mat2gray(img);  %任意区间映射到[0,1];
% % [m n dim]=size(img);
% % imshow(img);
% % %%图像的RGB
% % R=img(:,:,1);
% % G=img(:,:,2);
% % B=img(:,:,3);
% % 
% % %%RGB2YUV
% % Y=zeros(m,n);   %亮度
% % U=zeros(m,n);   %彩度
% % V=zeros(m,n);   %浓度
% % matrix=[0.299 0.587 0.114;
% %         -0.14713 -0.28886 0.436;
% %         0.615 -0.51498 -0.10001];
% % for i=1:m
% %    for j=1:n 
% %         tmp=matrix*[R(i,j) G(i,j) B(i,j)]';
% %         Y(i,j)=tmp(1);
% %         U(i,j)=tmp(2);
% %         V(i,j)=tmp(3);
% %    end
% % end

% % %%YUV2RGB
% % matrix=inv(matrix);
% % for i=1:m
% %    for j=1:n 
% %         tmp=matrix*[Y(i,j) U(i,j) V(i,j)]';
% %         R(i,j)=tmp(1);
% %         G(i,j)=tmp(2);
% %         B(i,j)=tmp(3);
% %    end
% % end
% % 
% % %%如果正反变换都没错的话，那么图像是不变的
% % img(:,:,1)=R;
% % img(:,:,2)=G;
% % img(:,:,3)=B;
% % figure;
% % imshow(img)



% % 
% % 
% % clear all;
% % close all;
% % clc;
% % 
img=imread('timg.jpg');
% img=mat2gray(img);  %任意区间映射到[0,1];


%%图像的RGB
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
[m n dim]=size(img);

%%RGB2YUV
Y=zeros(m,n);   %亮度
U=zeros(m,n);   %彩度
V=zeros(m,n);   %浓度
Y=0.299*R+ 0.587*G+ 0.114*B

U=-0.147*R-0.289*G+0.436*B
V=0.615*R-0.515*G-0.100*B

% % 
% % %I=rgb2gray(Y);  %必有不然图片会被压缩到一半大小
% % % %对于RGB图像必须做的一步，也可以用im2double函数
% % I=im2double(Y);
% %  F=fft2(I);          %傅里叶变换
% %  F1=log(abs(F)+1);   %取模并进行缩放
% % 
% %  Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
% %  S=log(abs(Fs)+1);    %取模并进行缩放
% % Y=S;
% %  imshow(Y,[])
 yuv = cat(3, Y, U, V);  
% %  rgb = cat(3, R, G, B);  
% %  
figure('NumberTitle', 'off', 'Name', 'YUV图像，Y通道，U通道，V通道 分别显示');%%figure改名字
subplot(221),imshow(yuv),title('yuv');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('U');
 subplot(224),imshow(V),title('V');
 
 RGB_ = ycbcr2rgb(yuv);%转成RGB
 figure
subplot(111),imshow(RGB_ ,[]),title('RGB');
 
figure('NumberTitle', 'off', 'Name', 'RGB图像和YUV图像对比');%%figure改名字
 subplot(121),imshow(RGB_),title('RGB');
  subplot(122),imshow(yuv),title('YUV');
% % 
% % figure('NumberTitle', 'off', 'Name', 'RGB图像，R通道，G通道，B通道分别显示');%%figure改名字
% % 
% %   subplot(221),imshow(rgb),title('rgb');
% % subplot(222),imshow(R),title('R');
% %  subplot(223),imshow(G),title('G');
% %  subplot(224),imshow(B),title('B');
 
% % % YUV2RGB
% % % matrix=inv(matrix);
% % % for i=1:m
% % %    for j=1:n 
% % %         tmp=matrix*[Y(i,j) U(i,j) V(i,j)]';
% % %         R(i,j)=tmp(1);
% % %         G(i,j)=tmp(2);
% % %         B(i,j)=tmp(3);
% % %    end
% % % end
% % % 
% % % %%%%如果正反变换都没错的话，那么图像是不变的
% % % img(:,:,1)=R;
% % % img(:,:,2)=G;
% % % img(:,:,3)=B;
% % % figure;
% % % imshow(img)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %  I=imread([pn fn ]);

% % RGB=imread('timg.jpg');%读入后便是一个RGB矩阵
% % YUV=rgb2ycbcr(RGB);%转成YUV
% %  RGB_ = ycbcr2rgb(YUV);%转成RGB
% %  figure('NumberTitle', 'off', 'Name', 'RGB');
% % subplot(131),imshow(RGB)  ,title('Y通道频谱图');
% % % % Y=imread([r g ]);
% % Y=YUV(:,:,1);
% % % % YUV(:,:,3)
% %  U=YUV(:,:,2);
% % V= YUV(:,:,3);
% % figure('NumberTitle', 'off', 'Name', 'Y');
% % subplot(131),imshow(Y,[])  ,title('Y');
% % subplot;
% %  A=imshow(P);
% % % % 对于RGB图像必须做的一步，也可以用im2double函数
%%Y=im2double(Y);%%%这个加上图会有错，会变暗

% %  F=fft2(Y);          %傅里叶变换
% %  % % F1=log(abs(F)+1);   %取模并进行缩放
% %  Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
% % % %   S=log(abs(Fs)+1);    %取模并进行缩放
% %   T=ifftshift(Fs);
% %   Y1=ifft2(T);
% %    figure('NumberTitle', 'off', 'Name', 'Y1');
% % subplot(131),imshow(Y1,[])  ,title('Y1');
% %   yuv=cat(3,Y1,U,V);
% %   RGB_ = ycbcr2rgb(yuv);%转成RGB
% %  figure('NumberTitle', 'off', 'Name', 'YUVfft');
% % subplot(131),imshow(S,[])  ,title('Y通道频谱图');
% %  subplot(132),imshow(yuv)  ,title('YUV');
% %   subplot(133),imshow(RGB_)  ,title('RGB');

% subplot(2,2,2);imshow(T1);title('低通滤波器');

% % RGB_ = ycbcr2rgb(YUV);%转成RGB
% % subplot
% %  imshow(RGB_);
 
 %RGB(:,:,1)为R分量矩阵
%RGB(:,:,2)为G分量矩阵
%RGB(:,:,3)为B分量矩阵
%YUV(:,:,1)为Y分量矩阵
%YUV(:,:,2)为U分量矩阵
%YUV(:,:,3)为V分量矩阵

%公式 Y = 0.2990*R + 0.5780*G + 0.1140*B + 0
%公式 U = 0.5000*R - 0.4187*G - 0.0813*B + 128
%公式 V = -0.1687*R - 0.3313*G + 0.5000*B + 128