clear
clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');

I=imread([pn fn ]);
figure;
subplot;
A=imshow(I);

I=rgb2gray(I);  %���в�ȻͼƬ�ᱻѹ����һ���С
%����RGBͼ���������һ����Ҳ������im2double����
%I=im2double(I);
F=fft2(I);          %����Ҷ�任
F1=log(abs(F)+1);   %ȡģ����������
% % subplot(2,2,2);imshow(T1);title('��ͨ�˲���');
Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
S=log(abs(Fs)+1);    %ȡģ����������

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
        
        h1=1/(1+0.414*(d/D1)^(2*n4));%�����ͨ�˲������ݺ���
              s1(i,j)=h1*Fs(i,j);
              T1(i, j) = h1;
        h0=1/(1+0.414*(d/D0)^(2*n4)); %�����ͨ�˲������ݺ���
               s0(i,j)=h0*Fs(i,j);
               T0(i, j) = h0;
        h2=1/(1+0.414*(d/D2)^(2*n4));%�����ͨ�˲������ݺ���
               s2(i,j)=h2*Fs(i,j);
               T2(i, j) = h2;
        h3=1/(1+0.414*(d/D3)^(2*n4)); %�����ͨ�˲������ݺ���
                s3(i,j)=h3*Fs(i,j);
                T3(i, j) = h3;
        h4=1/(1+0.414*(d/D4)^(2*n4)); %�����ͨ�˲������ݺ���
                s4(i,j)=h4*Fs(i,j);
                T4(i, j) = h4; 
        h5=1/(1+0.414*(d/D5)^(2*n4)); %�����ͨ�˲������ݺ���
                s5(i,j)=h5*Fs(i,j);
                T5(i, j) = h5;
                
                
% %         h4=1/(1+0.414*(d/D0)^(2*n4));
% %                s4(i,j)=h1*Fs(i,j);
% %                T4(i, j) = h1;

   end
end
fr0=real(ifft2(ifftshift(s0)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
fr1=real(ifft2(ifftshift(s1)));
fr2=real(ifft2(ifftshift(s2)));
fr3=real(ifft2(ifftshift(s3)));
fr4=real(ifft2(ifftshift(s4)));
fr5=real(ifft2(ifftshift(s5)));


R0=im2uint8(mat2gray(fr0));    %����ͼ������
R1=im2uint8(mat2gray(fr1)); 
R2=im2uint8(mat2gray(fr2)); 
R3=im2uint8(mat2gray(fr3)); 
R4=im2uint8(mat2gray(fr4));
R5=im2uint8(mat2gray(fr5));
figure
subplot(221);imshow(R0,[]);title('D0=10��Ч��ͼ');
subplot(222);imshow(T0);title('��ͨ�˲���D0=10');
subplot(223);imshow(R1,[]);title('D1=20��Ч��ͼ');
subplot(224);imshow(T1),title('��ͨ�˲���D1=20');
figure
subplot(221);imshow(R2,[]);title('D2=40��Ч��ͼ');
subplot(222);imshow(T2);title('��ͨ�˲���D2=40');
subplot(223);imshow(R3,[]);title('D3=80��Ч��ͼ');
subplot(224);imshow(T3),title('��ͨ�˲���D3=80');
figure
subplot(221);imshow(R4,[]);title('D4=160��Ч��ͼ');
subplot(222);imshow(T4),title('��ͨ�˲���D4=160');
subplot(223);imshow(R5,[]);title('D5=250��Ч��ͼ');
subplot(224);imshow(T5),title('��ͨ�˲���D3=250');



 %[M,N]=size(g);
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
