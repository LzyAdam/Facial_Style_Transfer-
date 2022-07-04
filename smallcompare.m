clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');

I=imread([pn fn ]);
figure;
subplot;
A=imshow(I);

I=rgb2gray(I);  %必有不然图片会被压缩到一半大小
%对于RGB图像必须做的一步，也可以用im2double函数
%I=im2double(I);
F=fft2(I);          %傅里叶变换
F1=log(abs(F)+1);   %取模并进行缩放
% % subplot(2,2,2);imshow(T1);title('低通滤波器');
Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
S=log(abs(Fs)+1);    %取模并进行缩放

n4=2;
D0=10;D1=20;
D2=40;D3=80;
D4=160;D5=250;
 [M,N]=size(F);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
        
        h1=1/(1+0.414*(d/D1)^(2*n4));%计算低通滤波器传递函数
              s1(i,j)=h1*Fs(i,j);
              T1(i, j) = h1;
        h0=1/(1+0.414*(d/D0)^(2*n4)); %计算低通滤波器传递函数
               s0(i,j)=h0*Fs(i,j);
               T0(i, j) = h0;
        h2=1/(1+0.414*(d/D2)^(2*n4));%计算低通滤波器传递函数
               s2(i,j)=h2*Fs(i,j);
               T2(i, j) = h2;
        h3=1/(1+0.414*(d/D3)^(2*n4)); %计算低通滤波器传递函数
                s3(i,j)=h3*Fs(i,j);
                T3(i, j) = h3;
        h4=1/(1+0.414*(d/D4)^(2*n4)); %计算低通滤波器传递函数
                s4(i,j)=h4*Fs(i,j);
                T4(i, j) = h4; 
        h5=1/(1+0.414*(d/D5)^(2*n4)); %计算低通滤波器传递函数
                s5(i,j)=h5*Fs(i,j);
                T5(i, j) = h5;
                
                
% %         h4=1/(1+0.414*(d/D0)^(2*n4));
% %                s4(i,j)=h1*Fs(i,j);
% %                T4(i, j) = h1;

   end
end
fr0=real(ifft2(ifftshift(s0)));  %频率域反变换到空间域，并取实部
fr1=real(ifft2(ifftshift(s1)));
fr2=real(ifft2(ifftshift(s2)));
fr3=real(ifft2(ifftshift(s3)));
fr4=real(ifft2(ifftshift(s4)));
fr5=real(ifft2(ifftshift(s5)));


R0=im2uint8(mat2gray(fr0));    %更改图像类型
R1=im2uint8(mat2gray(fr1)); 
R2=im2uint8(mat2gray(fr2)); 
R3=im2uint8(mat2gray(fr3)); 
R4=im2uint8(mat2gray(fr4));
R5=im2uint8(mat2gray(fr5));
figure
subplot(221);imshow(R0,[]);title('D0=10的效果图');
subplot(222);imshow(T0);title('低通滤波器D0=10');
subplot(223);imshow(R1,[]);title('D1=20的效果图');
subplot(224);imshow(T1),title('低通滤波器D1=20');
figure
subplot(221);imshow(R2,[]);title('D2=40的效果图');
subplot(222);imshow(T2);title('低通滤波器D2=40');
subplot(223);imshow(R3,[]);title('D3=80的效果图');
subplot(224);imshow(T3),title('低通滤波器D3=80');
figure
subplot(221);imshow(R4,[]);title('D4=160的效果图');
subplot(222);imshow(T4),title('低通滤波器D4=160');
subplot(223);imshow(R5,[]);title('D5=250的效果图');
subplot(224);imshow(T5),title('低通滤波器D3=250');



 %[M,N]=size(g);
% m=fix(M/2);
% n=fix(N/2);
% for i=1:M
%    for j=1:N
%         d=sqrt((i-m)^2+(j-n)^2);
% % h=1/(1+(d/D0)^(2*n));   
%         h1=1/(1+0.414*(d/w4)^(2*n4));%计算低通滤波器传递函数
%         s1(i,j)=h1*g(i,j);
%         T1(i, j) = h1;
% 
%    end
% end
