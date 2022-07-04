clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','选择图片');

I=imread([pn fn ]);
figure;
subplot;
A=imshow(I);
% % % 
% % % 
I=rgb2gray(I);  
%对于RGB图像必须做的一步，也可以用im2double函数
I=im2double(I);
F=fft2(I);          %傅里叶变换
F1=log(abs(F)+1);   %取模并进行缩放
% % subplot(2,2,2);imshow(T1);title('低通滤波器');
Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
S=log(abs(Fs)+1);    %取模并进行缩放

n4=2;
D0=80;D1=20;D2=40;D3=60;D4=80;
 [M,N]=size(F);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
       h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/D0)^(2*n4));%计算低通滤波器传递函数
        s1(i,j)=h1*Fs(i,j);
        T1(i, j) = h1;

   end
end
figure
subplot(2,2,1);imshow(I);title('原图');
subplot(2,2,3);imshow(S,[]);title('频移后的频谱图');

fr=real(ifft2(ifftshift(s1)));  %频率域反变换到空间域，并取实部
ret=im2uint8(mat2gray(fr));    %更改图像类型
subplot(2,2,4);imshow(ret),title('逆傅里叶变换');
subplot(2,2,2);imshow(T1);title('低通滤波器D0=10');
 figure
 subplot;imshow(ret);title('低通滤波器');


 [M,N]=size(g);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/w4)^(2*n4));%计算低通滤波器传递函数
        s1(i,j)=h1*g(i,j);
        T1(i, j) = h1;

   end
end

n4=2;w4=80;%ER阶巴特沃斯(Butterworth)低通滤波器,截止频率为80
f=im2double(I);
g=fft2(f);%傅立叶变换
g=fftshift(g);%转换数据矩阵
[M,N]=size(g);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
% h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/w4)^(2*n4));%计算低通滤波器传递函数
        s1(i,j)=h1*g(i,j);
        T1(i, j) = h1;

   end
end
y1=im2uint8(real(ifft2(ifftshift(s1))));
figure,subplot(2,2,1),mesh(T1),title('滤波器透视图');
subplot(2,2,2),imshow(T1),title('滤波器示意图');

% figure,subplot(2,2,1),imshow(y1),title('原图');
%subplot(2,2,4),imshow(h1),title('半径为5的BLPF滤波器');