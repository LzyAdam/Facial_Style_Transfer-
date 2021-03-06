
%%%%%%%%%%%%%%%高斯低通滤波%%%%%%%%%%%%%%%%


clear%%千万不能没有要不然，有些莫名其妙的错
 clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');
 
% RGB=imread('timg.jpg');
RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字
imshow(RGB),title('原图');
figure('NumberTitle', 'off', 'Name', '理想低通滤波器');%%figure改名字
imshow(RGB),title('原始图像');
%%%%%%image=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令

R=RGB(:,:,1);%为R分量矩阵
G=(RGB(:,:,2));%为G分量矩阵
B=(RGB(:,:,3));%为B分量矩阵




YUV=rgb2ycbcr(RGB);%转成YUV

%%%分三通道%%%%%%%%%%
Y=(YUV(:,:,1));%为Y分量矩阵
U=(YUV(:,:,2));%为U分量矩阵
V=(YUV(:,:,3));%为V分量矩阵


%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
yuv = cat(3, Y, U, V);  

%%%%%显示图像的八股文%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YUV图像，Y通道，U通道，V通道 分别显示');%%figure改名字
subplot(221),imshow(yuv),title('YUV');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('U');
 subplot(224),imshow(V),title('V');

 %Y=im2double(Y);
  F=fft2(Y);          %傅里叶变换
 F1=log(abs(F)+1);   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
Fs=fftshift(F);%% 我曹不能取模，取模他妈的出倒影 虚部去掉有问题
  %Fs=real(fftshift(F));      %将频谱图中零频率成分移动至频谱图中心
  S=log(abs(Fs)+1);    %取模并进行缩放
FFt= fftshift(F1);   %YUV后，Y通道的傅里叶频谱图，YUV图，RGB图NO real
  Y=S;
% % I_C_rep=im2uint8(real(ifft2(J)));
  yuv= cat(3, Y,U,V); 

 
  tic
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
           % D = sqrt((i-width/2)^2+(j-high/2)^2);%%高斯
        %%%%%%%%%%%%理想低通滤波器%%%%%%%%% 
        
    h0 = exp(-1/2*(d.^2)/(D0*D0));%%%%%高斯
        
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;滤波器处理过的图像
               T0(i, j) = h0;                           %%D0=10;滤波器的样子
          
             h1= exp(-1/2*(d.^2)/(D1*D1));%%%%%高斯
       
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;滤波器处理过的图像
              T1(i, j) = h1;                          %%%%D1=20;滤波器的样子
       
           h2 = exp(-1/2*(d.^2)/(D2*D2));%%%%%高斯
       
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;滤波器处理过的图像
               T2(i, j) = h2;                         %%%%D2=40;滤波器的样子
%                
               h3 = exp(-1/2*(d.^2)/(D3*D3));%%%%%高斯
        
                s3(i,j)=h3*Fs(i,j);                  %D3=60;滤波器处理过的图像
                T3(i, j) = h3;                         %%;D3=60;滤波器的样子
        
           h4 = exp(-1/2*(d.^2)/(D4*D4));%%%%%高斯
      
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;滤波器处理过的图像
                T4(i, j) = h4;                          %%D4=80;滤波器的样子
                 
               h5 = exp(-1/2*(d.^2)/(D5*D5));%%%%%高斯
         
                s5(i,j)=h5*Fs(i,j);                  %%D5=255;滤波器处理过的图像
                T5(i, j) = h5;                         %%D5=255;滤波器的样子
       
   end
end
toc


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
image=fr0+fr10+fr21+fr32+fr43+fr54;
figure('NumberTitle', 'off', 'Name', '高斯斯滤波器');
imshow(image,[]);title('最终效果');%%D0=10;滤波器处理过的图像

% toc
% % figure('NumberTitle', 'off', 'Name', '理想低通滤波演示图');
% figure('NumberTitle', 'off', 'Name', '高斯低通滤波演示图');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(T0);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr54,[]);title('D5=255 D5-D4');
% subplot(3,4,12);imshow(T5-T4),title('低通滤波器D5=255 D5-D4');
% % 
% % % 
% % % 
% %%%%%%%%%yuv lowpass filter result%%%%%%%%%%%
% % % % % % % %    rgb = cat(3, R, G, B);  
% yuv0 = cat(3, fr0, U, V);  
% yuv1= cat(3, fr10, U, V);  
% yuv2 = cat(3, fr21, U, V);  
% yuv3 = cat(3, fr32, U, V);  
% yuv4 = cat(3,  fr43,U, V);  
% yuv5 = cat(3, fr54,U,V);  
% 
% yuv0p= cat(3, fr0,U,V);  
% yuv1p= cat(3, fr1, U, V);  
% yuv2p= cat(3, fr2, U, V);  
% yuv3p = cat(3, fr3, U, V);  
% yuv4p= cat(3,  fr4,U, V);  
% yuv5p= cat(3, fr5,U,V);  
% figure('NumberTitle', 'off', 'Name', '=>YUV012345,YUV0p1p2p3p4p5p');
% subplot(3,4,1);imshow(yuv0,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(yuv0p,[]);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(yuv1p,[]),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(yuv2p);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(yuv3p,[]),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(yuv4p),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(yuv5p,[]),title('低通滤波器D5=255 D5-D4');
% 
% figure('NumberTitle', 'off', 'Name', '5=>YUV6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(yuv0,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');
% 
% %  figure('NumberTitle', 'off', 'Name', '55=>YUV6个不同频段滤波器样子及处理后图像');
% % subplot(3,4,1);imshow(yuv0P,[]);title('D0=10的效果图');
% % subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
% % subplot(3,4,3);imshow(yuv1P,[]);title('D1=20 D1-D0');
% % subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
% % subplot(3,4,5);imshow(yuv2P,[]);title('D2=40 D2-D1');
% % subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% % subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% % subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% % subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% % subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% % subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% % subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');
% 
%  RGB_0= ycbcr2rgb(yuv0);%转成RGB
%  RGB_1= ycbcr2rgb(yuv1);
%  RGB_2= ycbcr2rgb(yuv2);
%  RGB_3= ycbcr2rgb(yuv3);
%  RGB_4= ycbcr2rgb(yuv4);
%   RGB_5= ycbcr2rgb(yuv5);
%   rgb0= ycbcr2rgb(yuv0p);%转成RGB
%  rgb1= ycbcr2rgb(yuv1p);
%  rgb2= ycbcr2rgb(yuv2p);
%  rgb3= ycbcr2rgb(yuv3p);
%  rgb4= ycbcr2rgb(yuv4p);
%   rgb5= ycbcr2rgb(yuv5p);
%   figure('NumberTitle', 'off', 'Name', '=>RGB012345,RGB0p1p2p3p4p5p');
% subplot(3,4,1);imshow(RGB_0,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(rgb0,[]);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(RGB_1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(rgb1,[]),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(RGB_2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(rgb2);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(RGB_3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(rgb3,[]),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(RGB_4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(rgb4 ),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(RGB_5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(rgb5,[]),title('低通滤波器D5=255 D5-D4');
% 
% % % % % % %     K5 = imlincomb(0.5,k4,0.5,RGB_5);
%  RGB=ycbcr2rgb((0.35*yuv0+0.25*yuv1+0.2*yuv2+0.12*yuv3+0.08*yuv4+0.05*yuv5));%I,J是读入的两幅图像
%  RGB_01=imlincomb(1.2,RGB_0,1,RGB_1,0.8,RGB_2,0.4,RGB_3,0.2,RGB_4,0.1,RGB_5);
%    RGB_012=1*RGB_0+1*RGB_1+1*RGB_2+1*RGB_3+1*RGB_4+1*RGB_5;
% %   RGB_012=imadd(0.7*RGB_012,0.7*RGB_3);
%    RGB_0123=imlincomb(1,RGB_0,2,RGB_1,2,RGB_2,3,RGB_3,3,RGB_4,10,RGB_5);
% %    RGB_01234=imadd(0.3*RGB_0123,0.7*RGB_4);
% RGB_01234=imlincomb(1,RGB_0,1,RGB_1,1,RGB_2,1,RGB_3,1,RGB_4,1,RGB_5);
%       RGB_012345=imlincomb(1,RGB_0,0.99,RGB_1,0.95,RGB_2,0.94,RGB_3,0.9,RGB_4,0.88,RGB_5);
%     
%  figure('NumberTitle', 'off', 'Name', '6=>YUV还原RGB不同6频段滤波器样子及处理后图像');
%  
% 
%   subplot(231),imshow(RGB,[])  ,title('YUV2RGB');
%  subplot(232),imshow(RGB_01,[])  ,title('YUV2RGB');
%  subplot(233),imshow( RGB_012,[])  ,title('YUV2RGB');
%   subplot(234),imshow( RGB_0123,[])  ,title('YUV2RGB');
% subplot(235),imshow( RGB_01234,[])  ,title('YUV2RGB');
%  subplot(236),imshow(RGB_012345,[])  ,title('YUV2RGB');
% 
% 
% 
% 
% 
% 
% 
% % P0=sumsqr(F);%%%%RGB 是原始的图像
% %   B=sumsqr(A);
% %     disp(P0);
%   
% %     P1=sumsqr(fr0);  disp(fr0);
% %     A=P1/P0; disp(A);
% %  P1=sumsqr(s0(i,j));
% P2=sumsqr(real(Fs));  disp(Fs);
