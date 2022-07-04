clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
 clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
 tic
rgb=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', 'ԭͼ');%%figure������

% %  hsi = rgb2hsi(rgb) 
% %  im1=imread('duck.jpg'); 
% %  im3=im1; 
% % im1=im2double(im1); 
% % [m,n,q]=size(im1); 
% ��ȡͼ��� RGB 3��ͨ��
% R=im1(:,:,1); G=im1(:,:,2); B=im1(:,:,3); 
% [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������


% rgb=im2double(rgb); 
tic
hsi = rgb2hsi(rgb) ;%ת��YUV
toc                                                                           % % figure
   r = rgb(:, :, 1); 
g = rgb(:, :, 2); 
b = rgb(:, :, 3); 
rgb = hsi2rgb(hsi) ;
% % imshow(YUV);

figure('NumberTitle', 'off', 'Name', 'hsi-rgb');%%figure������
subplot(221),imshow(hsi),title('rgb2hsi');
subplot(222),imshow(rgb,[]),title('hsi-rgb');
 subplot(223),imshow(hsi),title('U');
 subplot(224),imshow(hsi),title('V');

% % %%%����ͨ��%%%%%%%%%%
% H = hsi(:, :, 1) * 2 * pi; ��ס���ܳ�2��
H = hsi(:, :, 1) ; 
S = hsi(:, :, 2); 
I = hsi(:, :, 3); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
hsi = cat(3, H, S, I); 
% %%%%%%��ʾͼ��İ˹���%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'HSIͼ��Hͨ����Sͨ����Iͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(hsi),title('HSI');
subplot(222),imshow(H,[]),title('H');
 subplot(223),imshow(S),title('S');
 subplot(224),imshow(I),title('I');
%  %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%���½��и���Ҷ�任%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(I);          %����Ҷ�任
%   F1=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
 Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ��Ҳ�Ծ���ֵ���ı���
%Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
                                        %%ע��Fs=F1,���fs=f,�ͳ�������
                                        %%%����Fs=F����6���˲����˳���ͼ�ɼ�����Fs=F1���ɼ�                         
%   S=log(abs(Fs)+1);    %ȡģ����������
%   FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
%   I=FFt;
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%��ԭ����YUV2RGB%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 I=(ifft2(ifftshift(Fs)));
hsi = cat(3, H, S, I); 
rgb = hsi2rgb(hsi) ;    %ת��RGB
 figure('NumberTitle', 'off', 'Name', 'HSIͼ��2RGB_');
 imshow(rgb);
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%�����Կ���������ԭRGBͼ��%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%dispaly the fft result%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'HSI��Iͨ���ĸ���ҶƵ��ͼ��HSIͼ��RGBͼ');
subplot(131),imshow(I,[])  ,title('Iͨ��Ƶ��ͼ');
subplot(132),imshow(hsi,[])  ,title('HSI=>FFT��ͨ��ͼ');
subplot(133),imshow(rgb,[])  ,title('RGBͼ');
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%��������ͨ�˲���������˹�˲�%%%%%%%%%%
%  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
%  
 n4=2;%step1%�˲����Ľ���2%%%%%%
 %%step2%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
%%%step3%%%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
 [M,N]=size(F);%%%%%�˲�����С������ͼƬ%%%%%%%
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);%%%%��㵽ͼ�����ľ���%%%%%%%  
        
        %%%%%%%%%%%%������˹��ͨ�˲���%%%%%%%%% 
        h0=1/(1+0.414*(d/D0)^(2*n4)); %����D0=10;��ͨ�˲������ݺ���
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;�˲����������ͼ��
               T0(i, j) = h0;                           %%D0=10;�˲���������
               
        h1=1/(1+0.414*(d/D1)^(2*n4));%�����ͨ�˲������ݺ���
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;�˲����������ͼ��
              T1(i, j) = h1;                          %%%%D1=20;�˲���������
       
        h2=1/(1+0.414*(d/D2)^(2*n4));%����D2=40;��ͨ�˲������ݺ���
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;�˲����������ͼ��
               T2(i, j) = h2;                         %%%%D2=40;�˲���������
               
        h3=1/(1+0.414*(d/D3)^(2*n4)); %����D3=60;��ͨ�˲������ݺ���
                 s3(i,j)=h3*Fs(i,j);                  %D3=60;�˲����������ͼ��
                T3(i, j) = h3;                         %%;D3=60;�˲���������
                
        h4=1/(1+0.414*(d/D4)^(2*n4)); %����D4=80;��ͨ�˲������ݺ���
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;�˲����������ͼ��
                T4(i, j) = h4;                          %%D4=80;�˲���������
                
        h5=1/(1+0.414*(d/D5)^(2*n4)); %����D5=255;��ͨ�˲������ݺ���
                s5(i,j)=h5*Fs(i,j);                  %%D5=255;�˲����������ͼ��
                T5(i, j) = h5;                         %%D5=255;�˲���������
       
   end
end

fr0=real(ifft2(ifftshift(s0)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
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
% figure('NumberTitle', 'off', 'Name', '6����ͬƵ���˲������Ӽ������ͼ��');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');%%D0=10;�˲����������ͼ��
% subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');%%D0=10;�˲�������
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;�˲����������ͼ��
% subplot(3,4,4);imshow(T1-T0),title('��ͨ�˲���D1=20 D1-D0');
% subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');
% hsi = cat(3, H, S, I); 
hsi0 = cat(3, H,S,fr0);  
hsi1= cat(3, H,S,fr10);  
hsi2 = cat(3, H,S,fr21);  
hsi3 = cat(3, H,S,fr32);  
hsi4 = cat(3,  H,S,fr43);  
hsi5 = cat(3, H,S,fr54);  
rgb0= hsi2rgb(hsi0);%ת��RGB
 rgb1= hsi2rgb(hsi1);
 rgb2= hsi2rgb(hsi2);
 rgb3= hsi2rgb(hsi3);
 rgb4= hsi2rgb(hsi4);
  rgb5= hsi2rgb(hsi5);
figure('NumberTitle', 'off', 'Name', 'HSI����6����ͬƵ���˲������Ӽ������ͼ��');
subplot(3,4,1);imshow(hsi0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(hsi1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(hsi2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(hsi3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(hsi4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(hsi5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
figure('NumberTitle', 'off', 'Name', 'HSI2RGB����6����ͬƵ���˲������Ӽ������ͼ��');
subplot(3,4,1);imshow(rgb0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(rgb1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(rgb2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(rgb3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(rgb4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(rgb5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');



rgb01=imadd(1*rgb0,0.9*rgb1);%I,J�Ƕ��������ͼ��
 rgb012=imadd(1*rgb01,0.8*rgb2);
  rgb0123=imadd(1*rgb012,0.7*rgb3);
   rgb01234=imadd(0.9*rgb0123,0.6*rgb4);
      rgb012345=imadd(1.2*rgb01234,1*rgb5);
   S=abs( rgb012345)+1;    %ȡģ����������
 figure('NumberTitle', 'off', 'Name', '6=>HSI��ԭRGBͼ��');
 


imshow(rgb012345,[])  ,title('6Ƶ�θ���');
 
  