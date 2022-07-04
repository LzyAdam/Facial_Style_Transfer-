clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');

I=imread([pn fn ]);
figure;
subplot;
A=imshow(I);

I=rgb2gray(I);  %必有不然图片会被压缩到一半大小
%对于RGB图像必须做的一步，也可以用im2double函数
I=im2double(I);
F=fft2(I);          %傅里叶变换
F1=log(abs(F)+1);   %取模并进行缩放
% % subplot(2,2,2);imshow(T1);title('低通滤波器');
Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
S=log(abs(Fs)+1);    %取模并进行缩放
%P=abs(F.^2); %求功率谱的绝对值
%P=imageperiodogram(I);

% % % % % % subplot(2,2,1);imshow(F,[]);
% % % % % figure
% % % % %  subplot(2,2,2); imshow(F1,[]);
% % % % % %subplot(2,2,3);imshow(Fs,[]);%没求实部所以显示不出来
% % % % %  subplot(2,2,4);imshow(S,[]);%不加“[]”显示不完全



n4=2;
D0=10;D1=20;
D2=40;D3=60;
D4=80;D5=255;
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
fr10=fr1-fr0;
fr2=real(ifft2(ifftshift(s2)));
fr21=fr2-fr1;
fr3=real(ifft2(ifftshift(s3)));
fr32=fr3-fr2;
fr4=real(ifft2(ifftshift(s4)));
fr43=fr4-fr3;
fr5=real(ifft2(ifftshift(s5)));
fr54=fr5-fr4;

% % R0=im2uint8(mat2gray(fr0));    %更改图像类型
% % R1=im2uint8(mat2gray(fr1)); 
% % R2=im2uint8(mat2gray(fr2)); 
% % R3=im2uint8(mat2gray(fr3)); 
% % R4=im2uint8(mat2gray(fr4));
% % R5=im2uint8(mat2gray(fr5));

figure
subplot(3,4,1);imshow(fr0,[]);title('D0=10的效果图');
subplot(3,4,2);imshow(T0);title('低通滤波器D0=10');
subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr54,[]);title('D5=2 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('低通滤波器D5=255 D5-D4');

% % E1=conv2(ifft(s0.^2),h0);




clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');

I1=imread([pn fn ]);
figure;
subplot;
A1=imshow(I1);

I1=rgb2gray(I1);  %必有不然图片会被压缩到一半大小
%对于RGB图像必须做的一步，也可以用im2double函数
%I=im2double(I);
Ft=fft2(I1);          %傅里叶变换
Ft1=log(abs(Ft)+1);   %取模并进行缩放
% % subplot(2,2,2);imshow(T1);title('低通滤波器');
Fs1=fftshift(Ft);      %将频谱图中零频率成分移动至频谱图中心
S1=log(abs(Fs1)+1);    %取模并进行缩放
%P=abs(F.^2); %求功率谱的绝对值
%P=imageperiodogram(I);




n4=2;
D0=10;D1=20;
D2=40;D3=60;
D4=80;D5=255;
 [M,N]=size(Ft);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
        
        h7=1/(1+0.414*(d/D1)^(2*n4));%计算低通滤波器传递函数
              s7(i,j)=h7*Fs1(i,j);
              T7(i, j) = h7;
        h6=1/(1+0.414*(d/D0)^(2*n4)); %计算低通滤波器传递函数
               s6(i,j)=h6*Fs1(i,j);
               T6(i, j) = h6;
        h8=1/(1+0.414*(d/D2)^(2*n4));%计算低通滤波器传递函数
               s8(i,j)=h8*Fs1(i,j);
               T8(i, j) = h8;
        h9=1/(1+0.414*(d/D3)^(2*n4)); %计算低通滤波器传递函数
                s9(i,j)=h9*Fs1(i,j);
                T9(i, j) = h9;
        h10=1/(1+0.414*(d/D4)^(2*n4)); %计算低通滤波器传递函数
                s10(i,j)=h10*Fs1(i,j);
                T10(i, j) = h10; 
        h11=1/(1+0.414*(d/D5)^(2*n4)); %计算低通滤波器传递函数
                s11(i,j)=h11*Fs1(i,j);
                T11(i, j) = h11;
                
                
% %         h4=1/(1+0.414*(d/D0)^(2*n4));
% %                s4(i,j)=h1*Fs(i,j);
% %                T4(i, j) = h1;

   end
end
fr6=real(ifft2(ifftshift(s6)));  %频率域反变换到空间域，并取实部
fr7=real(ifft2(ifftshift(s7)));
fr76=fr7-fr6;
fr8=real(ifft2(ifftshift(s8)));
fr87=fr8-fr7;
fr9=real(ifft2(ifftshift(s9)));
fr98=fr9-fr8;
fr10=real(ifft2(ifftshift(s10)));
fr109=fr10-fr9;
fr11=real(ifft2(ifftshift(s11)));
fr1110=fr11-fr10;

% % R0=im2uint8(mat2gray(fr0));    %更改图像类型
% % R1=im2uint8(mat2gray(fr1)); 
% % R2=im2uint8(mat2gray(fr2)); 
% % R3=im2uint8(mat2gray(fr3)); 
% % R4=im2uint8(mat2gray(fr4));
% % R5=im2uint8(mat2gray(fr5));

figure
subplot(3,4,1);imshow(fr6,[]);title('D0=10的效果图');
subplot(3,4,2);imshow(T6);title('低通滤波器D0=10');
subplot(3,4,3);imshow(fr76,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T7-T6),title('低通滤波器D1=20 D1-D0');
subplot(3,4,5);imshow(fr87,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T8-T7);title('低通滤波器D2=40 D2-D1');
subplot(3,4,7);imshow(fr98,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T9-T8),title('低通滤波器D3=60 D3-D2');
subplot(3,4,9);imshow(fr109,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T10-T9),title('低通滤波器D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr1110,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T11-T10),title('低通滤波器D5=255 D5-D4');



% % % %  [M,N]=size(F);%傅里叶变换图片的大小用MN去衡量
% % % % m=fix(M/2);%中点
% % % % n=fix(N/2);
% % % % for i=1:M
% % % %    for j=1:N
% % % %         d=sqrt((i-m)^2+(j-n)^2);%测距离画圆
% % % % E6=conv2(ifft(s6.^2),h6);
% % % % 
% % % % G1=sqrt(E1/(E6+0.00001));
% % % % I1new=ifft(s1)*G1;
% % % % imshow(I1);
% % % % 
