clear%%千万不能没有要不然，有些莫名其妙的错
 clc
% [R,G,B]=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令
clear%%千万不能没有要不然，有些莫名其妙的错
 clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');
 tic
 RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字


tic
YUV=rgb2ycbcr(RGB);%转成YUV
toc                                                              % % figure
                                                                            % % imshow(YUV);

%%%分三通道%%%%%%%%%%
Y=YUV(:,:,1);%为Y分量矩阵
U=YUV(:,:,2);%为U分量矩阵
V=YUV(:,:,3);%为V分量矩阵

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
yuv = cat(3, Y, U, V);  
%%%%%%显示图像的八股文%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCr图像，Y通道，Cb通道，Cr通道 分别显示');%%figure改名字
subplot(221),imshow(yuv),title('YCbCr');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
 %%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%以下进行傅里叶变换%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(Y);          %傅里叶变换
%   F1=real(log(abs(F)+1));   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
  Fs=fftshift(F);%% 我曹不能取模，取模他妈的出倒影，也对绝对值负的变正
  %Fs=real(fftshift(F));      %将频谱图中零频率成分移动至频谱图中心
                                        %%注意Fs=F1,如果fs=f,就出问题了
                                        %%%但是Fs=F后面6个滤波器滤出的图可见，用Fs=F1不可见                         
%   S=log(abs(Fs)+1);    %取模并进行缩放
%   FFt= real(fftshift(F1));   %YUV后，Y通道的傅里叶频谱图，YUV图，RGB图
%   Y=FFt;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%还原测试YUV2RGB%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=ifft2(ifftshift(Fs));
 yuv= cat(3, Y,U,V); 
 RGB_ = ycbcr2rgb(YUV);%转成RGB
 figure('NumberTitle', 'off', 'Name', 'YCbCr图像2RGB_');
 imshow(RGB_);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%经测试可以完美还原RGB图像%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%dispaly the fft result%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCr后，Y通道的傅里叶频谱图，YCbCr图，RGB图');
subplot(131),imshow(Y,[])  ,title('Y通道频谱图');
subplot(132),imshow(yuv,[])  ,title('YCbCr=>FFT三通道图');
subplot(133),imshow(RGB_,[])  ,title('RGB图');
%%%%%%%dispaly the fft result%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%下面代码低通滤波器巴特沃斯滤波%%%%%%%%%%
 %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
 
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
figure('NumberTitle', 'off', 'Name', '6个不同频段滤波器样子及处理后图像');
subplot(3,4,1);imshow(fr0,[]);title('D0=10的效果图');%%D0=10;滤波器处理过的图像
subplot(3,4,2);imshow(T0);title('低通滤波器D0=10');%%D0=10;滤波器样子
subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;滤波器处理过的图像
subplot(3,4,4);imshow(T1-T0),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('低通滤波器D5=255 D5-D4');

yuv0 = cat(3, fr0,U,V);  
yuv1= cat(3, fr10, U, V);  
yuv2 = cat(3, fr21, U, V);  
yuv3 = cat(3, fr32, U, V);  
yuv4 = cat(3,  fr43,U, V);  
yuv5 = cat(3, fr54,U,V);  
RGB_0= ycbcr2rgb(yuv0);%转成RGB
 RGB_1= ycbcr2rgb(yuv1);
 RGB_2= ycbcr2rgb(yuv2);
 RGB_3= ycbcr2rgb(yuv3);
 RGB_4= ycbcr2rgb(yuv4);
  RGB_5= ycbcr2rgb(yuv5);
figure('NumberTitle', 'off', 'Name', 'YCbCr复合6个不同频段滤波器样子及处理后图像');
subplot(3,4,1);imshow(yuv0,[]);title('D0=10的效果图');
subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J是读入的两幅图像
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(1*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
       S=abs( RGB_012345)+1;    %取模并进行缩放
 figure('NumberTitle', 'off', 'Name', '6=>YCbCr还原RGB后图像');
 


imshow(RGB_012345,[])  ,title('6频段复合');






