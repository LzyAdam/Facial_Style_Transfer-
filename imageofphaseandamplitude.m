[fn,pn,fi]=uigetfile('*.bmp','ѡ��ͼƬ');
I=imshow([pn fn ]);
%subplot(221);
%b=fft2(I);
%c=imshow(b);
I=imread([pn fn ]);
I=rgb2gray(I);
I=im2double(I);
F=fft2(I);
F=fftshift(F);
F=abs(F);
T=log(F+1);
figure;

A=imshow(T,[]);
%subplot(222);

clc;%��������λ��ͼ�Ĵ���
img=imread('15.bmp');
%img=double(img);
f=fft2(img);        %����Ҷ�任
f=fftshift(f);      %ʹͼ��Գ�
r=real(f);          %ͼ��Ƶ��ʵ��
i=imag(f);          %ͼ��Ƶ���鲿
margin=log(abs(f));      %ͼ������ף���log������ʾ
phase=log(angle(f)*180/pi);     %ͼ����λ��
l=log(f);           
subplot(2,2,1),imshow(img),title('Դͼ��');
%subplot(2,2,2),imshow(l,[]),title('ͼ��Ƶ��');
subplot(2,2,3),imshow(margin,[]),title('ͼ�������');
subplot(2,2,4),imshow(phase,[]),title('ͼ����λ��');


