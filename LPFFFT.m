clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');

I=imread([pn fn ]);
figure;
subplot;
A=imshow(I);
% % % 
% % % 
I=rgb2gray(I);  
%����RGBͼ���������һ����Ҳ������im2double����
I=im2double(I);
F=fft2(I);          %����Ҷ�任
F1=log(abs(F)+1);   %ȡģ����������
% % subplot(2,2,2);imshow(T1);title('��ͨ�˲���');
Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
S=log(abs(Fs)+1);    %ȡģ����������

n4=2;
D0=80;D1=20;D2=40;D3=60;D4=80;
 [M,N]=size(F);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
       h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/D0)^(2*n4));%�����ͨ�˲������ݺ���
        s1(i,j)=h1*Fs(i,j);
        T1(i, j) = h1;

   end
end
figure
subplot(2,2,1);imshow(I);title('ԭͼ');
subplot(2,2,3);imshow(S,[]);title('Ƶ�ƺ��Ƶ��ͼ');

fr=real(ifft2(ifftshift(s1)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
ret=im2uint8(mat2gray(fr));    %����ͼ������
subplot(2,2,4);imshow(ret),title('�渵��Ҷ�任');
subplot(2,2,2);imshow(T1);title('��ͨ�˲���D0=10');
 figure
 subplot;imshow(ret);title('��ͨ�˲���');


 [M,N]=size(g);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/w4)^(2*n4));%�����ͨ�˲������ݺ���
        s1(i,j)=h1*g(i,j);
        T1(i, j) = h1;

   end
end

n4=2;w4=80;%ER�װ�����˹(Butterworth)��ͨ�˲���,��ֹƵ��Ϊ80
f=im2double(I);
g=fft2(f);%����Ҷ�任
g=fftshift(g);%ת�����ݾ���
[M,N]=size(g);
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);
% h=1/(1+(d/D0)^(2*n));   
        h1=1/(1+0.414*(d/w4)^(2*n4));%�����ͨ�˲������ݺ���
        s1(i,j)=h1*g(i,j);
        T1(i, j) = h1;

   end
end
y1=im2uint8(real(ifft2(ifftshift(s1))));
figure,subplot(2,2,1),mesh(T1),title('�˲���͸��ͼ');
subplot(2,2,2),imshow(T1),title('�˲���ʾ��ͼ');

% figure,subplot(2,2,1),imshow(y1),title('ԭͼ');
%subplot(2,2,4),imshow(h1),title('�뾶Ϊ5��BLPF�˲���');