 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');

 I=imread([pn fn ]);
I=rgb2gray(I);
A=imshow(I);
I=im2double(I);
[M,N]=size(I);
M0=M/2;
N0=N/2;
J=fft2(I);
%J=log(abs(J)+1);���Ū������ifft�ͺ����ˣ����Ӧ���Ǽ��ô�С�ģ�
%J=real(J);

J_shift=fftshift(J);
%J_shift=log(abs(J_shift)+1); 
n=2;
D0=20;
for x=1:M    
    for y=1:N        %����㣨x,y�������ĵ�ľ���         
          d=sqrt((x-M0)^2+(y-N0)^2);        %���������˹�˲���                
          h=1/(1+(d/D0)^(2*n));        %���˲�������������                         
          J_shift(x,y)=J_shift(x,y)*h;    
        end
   end 
J=real(ifftshift(J_shift));
I_D_rep=real(ifft2(J));
I_C_rep=im2uint8(real(ifft2(J)));

figure;
subplot(221);     imshow(I);           title('Ըͼ��');
subplot(222);      imshow(I_D_rep,[]);      title('ifft2')
subplot(223);     imshow(h,[]);           title('lvboq');
subplot(224);      imshow(J_shift,[]);    title('Ƶ��')
figure;
imshow(I_C_rep,[]);
title('3ת��unit8');
figure
imshow(J_shift,[]);
% 
% 

% n4=2;w4=80;%ER�װ�����˹(Butterworth)��ͨ�˲���,��ֹƵ��Ϊ80
% f=im2double(I);
% g=fft2(f);%����Ҷ�任
% g=fftshift(g);%ת�����ݾ���
% [M,N]=size(g);
% m=fix(M/2);
% n=fix(N/2);
% for i=1:M
%    for j=1:N
%         d=sqrt((i-m)^2+(j-n)^2);
% % h=1/(1+(d/D0)^(2*n));   
%         h1=1/(1+0.414*(d/w4)^(2*n4));%�����ͨ�˲������ݺ���
%         s1(i,j)=h1*g(i,j);
%         T1(i, j) = h1;
% 
%    end
% end
% y1=im2uint8(real(ifft2(ifftshift(s1))));
% figure,subplot(1,2,1),mesh(T1),title('�˲���͸��ͼ');
% subplot(1,2,2),imshow(T1),title('�˲���ʾ��ͼ');
% 
% figure,subplot(3,2,1),imshow(I),title('ԭͼ');
% subplot(3,2,2),imshow(y1),title('�뾶Ϊ5��BLPF�˲���');

