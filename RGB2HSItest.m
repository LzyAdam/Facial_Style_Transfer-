clear%%千万不能没有要不然，有些莫名其妙的错
 clc
 clear%%千万不能没有要不然，有些莫名其妙的错
 clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');
 tic
rgb=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字

% %  hsi = rgb2hsi(rgb) 
% %  im1=imread('duck.jpg'); 
% %  im3=im1; 
% % im1=im2double(im1); 
% % [m,n,q]=size(im1); 
% 获取图像的 RGB 3个通道
% R=im1(:,:,1); G=im1(:,:,2); B=im1(:,:,3); 
% [R,G,B]=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令


% rgb=im2double(rgb); 
tic
hsi = rgb2hsi(rgb) ;%转成YUV
toc                                                                           % % figure
   r = rgb(:, :, 1); 
g = rgb(:, :, 2); 
b = rgb(:, :, 3); 
rgb = hsi2rgb(hsi) ;
% % imshow(YUV);

figure('NumberTitle', 'off', 'Name', 'hsi-rgb');%%figure改名字
subplot(221),imshow(hsi),title('rgb2hsi');
subplot(222),imshow(rgb,[]),title('hsi-rgb');
 subplot(223),imshow(hsi),title('U');
 subplot(224),imshow(hsi),title('V');

% % %%%分三通道%%%%%%%%%%
% H = hsi(:, :, 1) * 2 * pi; 记住不能乘2π
H = hsi(:, :, 1) ; 
S = hsi(:, :, 2); 
I = hsi(:, :, 3); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
hsi = cat(3, H, S, I); 
% %%%%%%显示图像的八股文%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'HSI图像，H通道，S通道，I通道 分别显示');%%figure改名字
subplot(221),imshow(hsi),title('HSI');
subplot(222),imshow(H,[]),title('H');
 subplot(223),imshow(S),title('S');
 subplot(224),imshow(I),title('I');
%  %%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%以下进行傅里叶变换%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(I);          %傅里叶变换
%   F1=real(log(abs(F)+1));   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
 Fs=fftshift(F);%% 我曹不能取模，取模他妈的出倒影，也对绝对值负的变正
%Fs=real(fftshift(F));      %将频谱图中零频率成分移动至频谱图中心
                                        %%注意Fs=F1,如果fs=f,就出问题了
                                        %%%但是Fs=F后面6个滤波器滤出的图可见，用Fs=F1不可见                         
%   S=log(abs(Fs)+1);    %取模并进行缩放
%   FFt= real(fftshift(F1));   %YUV后，Y通道的傅里叶频谱图，YUV图，RGB图
%   I=FFt;
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%还原测试YUV2RGB%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 I=(ifft2(ifftshift(Fs)));
hsi = cat(3, H, S, I); 
rgb = hsi2rgb(hsi) ;    %转成RGB
 figure('NumberTitle', 'off', 'Name', 'HSI图像2RGB_');
 imshow(rgb);
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%经测试可以完美还原RGB图像%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%dispaly the fft result%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'HSI后，I通道的傅里叶频谱图，HSI图，RGB图');
subplot(131),imshow(I,[])  ,title('I通道频谱图');
subplot(132),imshow(hsi,[])  ,title('HSI=>FFT三通道图');
subplot(133),imshow(rgb,[])  ,title('RGB图');
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%下面代码低通滤波器巴特沃斯滤波%%%%%%%%%%
%  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
%  
 n4=2;%step1%滤波器的阶数2%%%%%%
 %%step2%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
%%%step3%%%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
 [M,N]=size(F);%%%%%滤波器大小适配与图片%%%%%%%
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);%%%%算点到图像中心距离%%%%%%%  
        
        %%%%%%%%%%%%巴特沃斯低通滤波器%%%%%%%%% 
        h0=1/(1+0.414*(d/D0)^(2*n4)); %计算D0=10;低通滤波器传递函数
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;滤波器处理过的图像
               T0(i, j) = h0;                           %%D0=10;滤波器的样子
               
        h1=1/(1+0.414*(d/D1)^(2*n4));%计算低通滤波器传递函数
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;滤波器处理过的图像
              T1(i, j) = h1;                          %%%%D1=20;滤波器的样子
       
        h2=1/(1+0.414*(d/D2)^(2*n4));%计算D2=40;低通滤波器传递函数
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;滤波器处理过的图像
               T2(i, j) = h2;                         %%%%D2=40;滤波器的样子
               
        h3=1/(1+0.414*(d/D3)^(2*n4)); %计算D3=60;低通滤波器传递函数
                 s3(i,j)=h3*Fs(i,j);                  %D3=60;滤波器处理过的图像
                T3(i, j) = h3;                         %%;D3=60;滤波器的样子
                
        h4=1/(1+0.414*(d/D4)^(2*n4)); %计算D4=80;低通滤波器传递函数
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;滤波器处理过的图像
                T4(i, j) = h4;                          %%D4=80;滤波器的样子
                
        h5=1/(1+0.414*(d/D5)^(2*n4)); %计算D5=255;低通滤波器传递函数
                s5(i,j)=h5*Fs(i,j);                  %%D5=255;滤波器处理过的图像
                T5(i, j) = h5;                         %%D5=255;滤波器的样子
       
   end
end

fr0=real(ifft2(ifftshift(s0)));  %频率域反变换到空间域，并取实部
fr1=real(ifft2(ifftshift(s1)));
fr10=fr1-fr0;
fr2=real(ifft2(ifftshift(s2)));
fr21=fr2-fr1;
fr3=real(ifft2(ifftshift(s3)));
fr32=fr3-fr2;
fr4=real(ifft2(ifftshift(s4)));
fr43=fr4-fr3;
fr5=real(ifft2(ifftshift(s5)));
fr54=fr5-fr4;
% figure('NumberTitle', 'off', 'Name', '6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10的效果图');%%D0=10;滤波器处理过的图像
% subplot(3,4,2);imshow(T0);title('低通滤波器D0=10');%%D0=10;滤波器样子
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;滤波器处理过的图像
% subplot(3,4,4);imshow(T1-T0),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4),title('低通滤波器D5=255 D5-D4');
% hsi = cat(3, H, S, I); 
hsi0 = cat(3, H,S,fr0);  
hsi1= cat(3, H,S,fr10);  
hsi2 = cat(3, H,S,fr21);  
hsi3 = cat(3, H,S,fr32);  
hsi4 = cat(3,  H,S,fr43);  
hsi5 = cat(3, H,S,fr54);  
rgb0= hsi2rgb(hsi0);%转成RGB
 rgb1= hsi2rgb(hsi1);
 rgb2= hsi2rgb(hsi2);
 rgb3= hsi2rgb(hsi3);
 rgb4= hsi2rgb(hsi4);
  rgb5= hsi2rgb(hsi5);
figure('NumberTitle', 'off', 'Name', 'HSI复合6个不同频段滤波器样子及处理后图像');
subplot(3,4,1);imshow(hsi0,[]);title('D0=10的效果图');
subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
subplot(3,4,3);imshow(hsi1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(hsi2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(hsi3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(hsi4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(hsi5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');
figure('NumberTitle', 'off', 'Name', 'HSI2RGB复合6个不同频段滤波器样子及处理后图像');
subplot(3,4,1);imshow(rgb0,[]);title('D0=10的效果图');
subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
subplot(3,4,3);imshow(rgb1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(rgb2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(rgb3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(rgb4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(rgb5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');



rgb01=imadd(1*rgb0,0.9*rgb1);%I,J是读入的两幅图像
 rgb012=imadd(1*rgb01,0.8*rgb2);
  rgb0123=imadd(1*rgb012,0.7*rgb3);
   rgb01234=imadd(0.9*rgb0123,0.6*rgb4);
      rgb012345=imadd(1.2*rgb01234,1*rgb5);
   S=abs( rgb012345)+1;    %取模并进行缩放
 figure('NumberTitle', 'off', 'Name', '6=>HSI还原RGB图像');
 


imshow(rgb012345,[])  ,title('6频段复合');
 
  