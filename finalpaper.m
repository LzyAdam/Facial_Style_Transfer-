% clear all;
% close all;
% clc;
% 
% img=imread('timg.jpg');
% %    img=mat2gray(img);  %��������ӳ�䵽[0,1];
% 
% 
% 
% R=img(:,:,1);                        %%ͼ���RGB
% G=img(:,:,2);
% B=img(:,:,3);
% [m n dim]=size(img);
% 
%RGB2YUV
% Y=zeros(m,n);   %����
% U=zeros(m,n);   %�ʶ�
% V=zeros(m,n);   %Ũ��
% Y=0.299.*R+0.578.*G+0.114.*B
% U=-0.1687.*R-0.3313.*G+0.5.*B+128
% V=0.5.*R-0.4187.*G-0.0813.*B+128
% yuv = cat(3, Y, U, V);  
% figure('NumberTitle', 'off', 'Name', 'YUVͼ��');
% imshow(yuv,[]);title('rgb2yuv');
% 
% RGB_ = ycbcr2rgb(yuv);%ת��RGB
% figure('NumberTitle', 'off', 'Name', 'RGBͼ��');
%  imshow(RGB_);title('yuv2rgb');
%  R= Y + 1.402 *(V-128);
% 
% G= Y - 0.34414* (U-128) - 0.71414 *(V-128);
% 
% B= Y + 1.772 *(U-128);
% % % R= 1.164*(Y-16) + 1.159*(V-128); 
% % % G = 1.164*(Y-16) - 0.380*(U-128) - 0.813*(V-128); 
% % % B = 1.164*(Y-16) + 2.018*(U-128);
% rgb=cat(3,R,G,B);
% figure
% imshow(rgb,[]);


% % I=rgb2gray(Y);  %���в�ȻͼƬ�ᱻѹ����һ���С
%����RGBͼ���������һ����Ҳ������im2double����
% I=im2double(Y);
%  F=fft2(I);          %����Ҷ�任
%  F1=log(abs(F)+1);   %ȡģ����������
% 
%  Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
%  S=log(abs(Fs)+1);    %ȡģ����������
% % % % % Y=S;
% % % % %  imshow(Y,[])
% % % % % yuv = cat(3, Y, U, V);  
% % % % %  rgb = cat(3, R, G, B);  
% 
%  
%  n4=2;
% D0=10;D1=20;
% D2=40;D3=60;
% D4=80;D5=255;
%  [M,N]=size(F);
% m=fix(M/2);
% n=fix(N/2);
% for i=1:M
%    for j=1:N
%         d=sqrt((i-m)^2+(j-n)^2);
%         
%         h1=1/(1+0.414*(d/D1)^(2*n4));%�����ͨ�˲������ݺ���
%               s1(i,j)=h1*Fs(i,j);
%               T1(i, j) = h1;
%         h0=1/(1+0.414*(d/D0)^(2*n4)); %�����ͨ�˲������ݺ���
%                s0(i,j)=h0*Fs(i,j);
%                T0(i, j) = h0;
%         h2=1/(1+0.414*(d/D2)^(2*n4));%�����ͨ�˲������ݺ���
%                s2(i,j)=h2*Fs(i,j);
%                T2(i, j) = h2;
%         h3=1/(1+0.414*(d/D3)^(2*n4)); %�����ͨ�˲������ݺ���
%                 s3(i,j)=h3*Fs(i,j);
%                 T3(i, j) = h3;
%         h4=1/(1+0.414*(d/D4)^(2*n4)); %�����ͨ�˲������ݺ���
%                 s4(i,j)=h4*Fs(i,j);
%                 T4(i, j) = h4; 
%         h5=1/(1+0.414*(d/D5)^(2*n4)); %�����ͨ�˲������ݺ���
%                 s5(i,j)=h5*Fs(i,j);
%                 T5(i, j) = h5;
%        
%    end
% end
% fr0=real(ifft2(ifftshift(s0)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
% fr1=real(ifft2(ifftshift(s1)));
% fr10=fr1-fr0;
% fr2=real(ifft2(ifftshift(s2)));
% fr21=fr2-fr1;
% fr3=real(ifft2(ifftshift(s3)));
% fr32=fr3-fr2;
% fr4=real(ifft2(ifftshift(s4)));
% fr43=fr4-fr3;
% fr5=real(ifft2(ifftshift(s5)));
% fr54=fr5-fr4;
% 
% 
% 
% % % % % Y=S;
% % % % %  imshow(Y,[])
% 
% % % % % % %   rgb = cat(3, R, G, B);  
% yuv0 = cat(3, fr0,U,V);  
% yuv1= cat(3, fr10, U, V);  
% yuv2 = cat(3, fr21, U, V);  
% yuv3 = cat(3, fr32, U, V);  
% yuv4 = cat(2,  U, V);  
% yuv5 = cat(3, fr54,U,V);  
% figure
% subplot(2,3,1);imshow(yuv0,[]);title('vic0')
% subplot(2,3,2);imshow(yuv1,[]);title('vic1')
% subplot(2,3,3);imshow(yuv2,[]);title('vic2')
% subplot(2,3,4);imshow(yuv3,[]);title('vic3')
% subplot(2,3,5);imshow(yuv4,[]);title('vic4')
% subplot(2,3,6);imshow(yuv5,[]);title('vic5')
% figure
% subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');
% subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0),title('��ͨ�˲���D1=20 D1-D0');
% subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');
% 
% yuv0 = cat(3, fr0,U,V); 
% % % r = 1.164*(y-16) + 1.159*(v-128); 
% % % g = 1.164*(y-16) - 0.380*(u-128) - 0.813*(v-128); 
% % % b = 1.164*(y-16) + 2.018*(u-128);
% 
% 
% % % figure('NumberTitle', 'off', 'Name', 'YUVͼ��Yͨ����Uͨ����Vͨ�� �ֱ���ʾ');%%figure������
% % % subplot(221),imshow(yuv),title('yuv');
% % % subplot(222),imshow(Y,[]),title('Y');
% % %  subplot(223),imshow(U),title('U');
% % %  subplot(224),imshow(V),title('V');
% % % figure('NumberTitle', 'off', 'Name', 'RGBͼ���YUVͼ��Ա�');%%figure������
% % %  subplot(121),imshow(rgb),title('RGB');
% % %   subplot(122),imshow(yuv),title('YUV');
% % % 
% % % figure('NumberTitle', 'off', 'Name', 'RGBͼ��Rͨ����Gͨ����Bͨ���ֱ���ʾ');%%figure������
% % % 
% % %   subplot(221),imshow(rgb),title('rgb');
% % % subplot(222),imshow(R),title('R');
% % %  subplot(223),imshow(G),title('G');
% % %  subplot(224),imshow(B),title('B');
%  �Լ�д�Ĺ�ʽת���������ס�%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%RGB2YUV������Դ���������
  [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������

YUV=rgb2ycbcr(RGB);%ת��YUV
%YUV=mat2gray(YUV); 
%%%����ͨ��%%%%%%%%%%
Y=YUV(:,:,1)%ΪY��������
U=YUV(:,:,2)%ΪU��������
V=YUV(:,:,3)%ΪV��������


%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv = cat(3, Y, U, V);  

%%%%%%��ʾͼ��İ˹���%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YUVͼ��Yͨ����Uͨ����Vͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('YUV');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('U');
 subplot(224),imshow(V),title('V');
 
 %%%%%%%fft transform%%%%%%%%%%%%%
 %%%%I=im2double(Y);!!!!!!�������ͼ����ɺں�ɫ
  F=fft2(Y);          %����Ҷ�任
 F1=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ

  Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
  S=log(abs(Fs)+1);    %ȡģ����������
 %FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
  Y=S;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%���������ʾ���%%%dispaly the fft result%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 yuv= cat(3, Y,U,V); 
 RGB_ = ycbcr2rgb(YUV);%ת��RGB
 figure
 imshow(RGB_);
figure('NumberTitle', 'off', 'Name', '3YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼFFT��RGBͼ');
subplot(131),imshow(Y,[])  ,title('Yͨ��Ƶ��ͼ');
subplot(132),imshow(yuv,[])  ,title('YUV����FFT�仯�����ͨ���ϳ�ͼ');
subplot(133),imshow(RGB_,[])  ,title('RGBͼ');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%������δ���%%%%%%%%%%��ʾ���%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%��������ͨ�˲���������˹�˲�%%%%%%%%%%
 %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
figure('NumberTitle', 'off', 'Name', '4=>Yͨ��Ƶ�װ�����˹�˲���ʾͼ');
subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');
% % 
% % 
% % 
%%%%%%%%%yuv lowpass filter result%%%%%%%%%%%
% % % % % % %    rgb = cat(3, R, G, B);  


%R0=im2uint8(mat2gray(fr0));    %����ͼ������
% % % R1=im2uint8(mat2gray(fr1)); 
% % % R2=im2uint8(mat2gray(fr2)); 
% % % R3=im2uint8(mat2gray(fr3)); 
% % % R4=im2uint8(mat2gray(fr4));
% % % R5=im2uint8(mat2gray(fr5));
% 
yuv0 = cat(3, fr0,U,V);  
yuv1= cat(3, fr10, U, V);  
yuv2 = cat(3, fr21, U, V);  
yuv3 = cat(3, fr32, U, V);  
yuv4 = cat(3,  fr43,U, V);  
yuv5 = cat(3, fr54,U,V);  

yuv0P = cat(3, fr0,U,V);  
yuv1P= cat(3, fr1, U, V);  
yuv2P= cat(3, fr2, U, V);  
yuv3P = cat(3, fr3, U, V);  
yuv4P = cat(3,  fr4,U, V);  
yuv5P = cat(3, fr5,U,V);  
figure('NumberTitle', 'off', 'Name', '5=>YUV6����ͬƵ���˲������Ӽ������ͼ��');
subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');

 

 RGB_0= ycbcr2rgb(yuv0P);%ת��RGB
 RGB_1= ycbcr2rgb(yuv1P);
 RGB_2= ycbcr2rgb(yuv2P);
 RGB_3= ycbcr2rgb(yuv3P);
 RGB_4= ycbcr2rgb(yuv4P);
  RGB_5= ycbcr2rgb(yuv5P);
  
  K1= imlincomb(0.5,RGB_0,0.5,RGB_1);
    K2 = imlincomb(0.5,K1,0.5,RGB_2);
    K3=imlincomb(0.5,K2,0.5,RGB_3);
%  K4= imlincomb(0.5,k3,0.5,RGB_4);
%     K5 = imlincomb(0.5,k4,0.5,RGB_5);
  
 RGB_01=imadd(RGB_0,RGB_1);%I,J�Ƕ��������ͼ��
 RGB_012=imadd(RGB_01,RGB_2);
  RGB_0123=imadd(RGB_012,RGB_3);
   RGB_01234=imadd(RGB_0123,RGB_4);
      RGB_012345=imadd(RGB_01234,RGB_5);
       S=abs( RGB_012345)+1;    %ȡģ����������
 figure('NumberTitle', 'off', 'Name', '6=>YUV��ԭRGB��ͬ6Ƶ���˲������Ӽ������ͼ��');
 
 subplot(231),imshow(RGB_012345,[])  ,title('YUV2RGB');
 subplot(232),imshow(K1,[])  ,title('YUV2RGB');
 subplot(233),imshow(K2,[])  ,title('YUV2RGB');
 subplot(234),imshow( RGB_3,[])  ,title('YUV2RGB');
 subplot(235),imshow( RGB_4,[])  ,title('YUV2RGB');
 subplot(236),imshow( RGB_5,[])  ,title('YUV2RGB');
% % % % % % % % % % yuvT1= cat(3, T1, U, V); 
% % % % % % % % % % yuvT10=yuvT1-yuvT0;
% % % % % % % % % % yuvT2 = cat(3, T2, U, V);  
% % % % % % % % % % yuvT3 = cat(3, T3-T2, U, V);  
% % % % % % % % % % yuvT4 = cat(3, T4,U, V);  
% % % % % % % % % % yuvT5 = cat(3, T5,U,V); 

% % % 
% % 
% % 

% % % % û�õĴ���% % % % % % % % % % % % figure('NumberTitle', 'off', 'Name', 'Yͨ��Ƶ�װ�����˹�˲�YUV��ͨ���ϳ�ͼ');
% % % % % % % % % % % % % % % % subplot(2,3,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ')
% % % % % % % % % % % % % % % % subplot(2,3,2);imshow(yuv1,[]);title('D1=20 D1-D0')
% % % % % % % % % % % % % % % % subplot(2,3,3);imshow(yuv2,[]);title('D3=60 D3-D2')
% % % % % % % % % % % % % % % % subplot(2,3,4);imshow(yuv3,[]);title('D3=60 D3-D2')
% % % % % % % % % % % % % % % % subplot(2,3,5);imshow(yuv4,[]);title('D4=80 D4-D3')
% % % % % % % % % % % % % % % % subplot(2,3,6);imshow(yuv5,[]);title('D5=80 D5-D4')

% % 
% %   %  [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
% % % 
% % % I1=imread([pn fn ]);
% % % figure;
% % % subplot;
% % % A1=imshow(I1);
% % % 
% % 
% % % %����RGBͼ���������һ����Ҳ������im2double����
% % % %I=im2double(I);
% % % Ft=fft2(I1);          %����Ҷ�任
% % % Ft1=log(abs(Ft)+1);   %ȡģ����������
% % % % % subplot(2,2,2);imshow(T1);title('��ͨ�˲���');
% % % Fs1=fftshift(Ft);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
% % % S1=log(abs(Fs1)+1);    %ȡģ����������
% % % %P=abs(F.^2); %�����׵ľ���ֵ
% % % %P=imageperiodogram(I);
% % % 
% % % 
% % % 
% % % 
% % % n4=2;
% % % D0=10;D1=20;
% % % D2=40;D3=60;
% % % D4=80;D5=255;
% % %  [M,N]=size(Ft);
% % % m=fix(M/2);
% % % n=fix(N/2);
% % % for i=1:M
% % %    for j=1:N
% % %         d=sqrt((i-m)^2+(j-n)^2);
% % %         
% % %         h7=1/(1+0.414*(d/D1)^(2*n4));%�����ͨ�˲������ݺ���
% % %               s7(i,j)=h7*Fs1(i,j);
% % %               T7(i, j) = h7;
% % %         h6=1/(1+0.414*(d/D0)^(2*n4)); %�����ͨ�˲������ݺ���
% % %                s6(i,j)=h6*Fs1(i,j);
% % %                T6(i, j) = h6;
% % %         h8=1/(1+0.414*(d/D2)^(2*n4));%�����ͨ�˲������ݺ���
% % %                s8(i,j)=h8*Fs1(i,j);
% % %                T8(i, j) = h8;
% % %         h9=1/(1+0.414*(d/D3)^(2*n4)); %�����ͨ�˲������ݺ���
% % %                 s9(i,j)=h9*Fs1(i,j);
% % %                 T9(i, j) = h9;
% % %         h10=1/(1+0.414*(d/D4)^(2*n4)); %�����ͨ�˲������ݺ���
% % %                 s10(i,j)=h10*Fs1(i,j);
% % %                 T10(i, j) = h10; 
% % %         h11=1/(1+0.414*(d/D5)^(2*n4)); %�����ͨ�˲������ݺ���
% % %                 s11(i,j)=h11*Fs1(i,j);
% % %                 T11(i, j) = h11;
% % %                 
% % %                 
% % % % %         h4=1/(1+0.414*(d/D0)^(2*n4));
% % % % %                s4(i,j)=h1*Fs(i,j);
% % % % %                T4(i, j) = h1;
% % % 
% % %    end
% % % end
% % % fr6=real(ifft2(ifftshift(s6)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
% % % fr7=real(ifft2(ifftshift(s7)));
% % % fr76=fr7-fr6;
% % % fr8=real(ifft2(ifftshift(s8)));
% % % fr87=fr8-fr7;
% % % fr9=real(ifft2(ifftshift(s9)));
% % % fr98=fr9-fr8;
% % % fr10=real(ifft2(ifftshift(s10)));
% % % fr109=fr10-fr9;
% fr11=real(ifft2(ifftshift(s11)));
% fr1110=fr11-fr10;
% 
% % % R0=im2uint8(mat2gray(fr0));    %����ͼ������
% % % R1=im2uint8(mat2gray(fr1)); 
% % % R2=im2uint8(mat2gray(fr2)); 
% % % R3=im2uint8(mat2gray(fr3)); 
% % % R4=im2uint8(mat2gray(fr4));
% % % R5=im2uint8(mat2gray(fr5));
% 
% % figure
% % subplot(3,4,1);imshow(fr6,[]);title('D0=10��Ч��ͼ');
% % subplot(3,4,2);imshow(T6);title('��ͨ�˲���D0=10');
% % subplot(3,4,3);imshow(fr76,[]);title('D1=20 D1-D0');
% % subplot(3,4,4);imshow(T7-T6),title('��ͨ�˲���D1=20 D1-D0');
% % subplot(3,4,5);imshow(fr87,[]);title('D2=40 D2-D1');
% % subplot(3,4,6);imshow(T8-T7);title('��ͨ�˲���D2=40 D2-D1');
% % subplot(3,4,7);imshow(fr98,[]);title('D3=60 D3-D2');
% % subplot(3,4,8);imshow(T9-T8),title('��ͨ�˲���D3=60 D3-D2');
% % subplot(3,4,9);imshow(fr109,[]);title('D4=80 D4-D3');
% % subplot(3,4,10);imshow(T10-T9),title('��ͨ�˲���D4=60 D4-D3 ');
% % subplot(3,4,11);imshow(fr1110,[]);title('D5=80 D5-D4');
% % subplot(3,4,12);imshow(T11-T10),title('��ͨ�˲���D5=255 D5-D4');

%  yuv = cat(3, Y, U, V);  
%   rgb = cat(3, R, G, B);

  




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%yuv2rgb
RGB_ = ycbcr2rgb(YUV);%ת��RGB
figure
 imshow(RGB_);
r=RGB(:,:,1)%ΪR��������
g=RGB(:,:,2)%ΪG��������
b=RGB(:,:,3)%ΪB��������
y=YUV(:,:,1)%ΪY��������
u=YUV(:,:,2)%ΪU��������
v=YUV(:,:,3)%ΪV��������












% % 
% % %%%%%%ifft����Ҷ���仯
% %  [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
% % 
% %  img=imread([pn fn ]);
% % subplot(2,2,1);imshow(img);title('ԭͼ');
% % f=rgb2gray(img);    %����RGBͼ���������һ����Ҳ������im2double����
% % F=fft2(f);          %����Ҷ�任
% % F1=log(abs(F)+1);   %ȡģ����������
% % subplot(2,2,2);imshow(F1,[]);title('����Ҷ�任Ƶ��ͼ');
% % Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
% % S=log(abs(Fs)+1);    %ȡģ����������
% % subplot(2,2,3);imshow(S,[]);title('Ƶ�ƺ��Ƶ��ͼ');
% % fr=real(ifft2(ifftshift(Fs)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
% % ret=im2uint8(mat2gray(fr));    %����ͼ������
% % subplot(2,2,4);imshow(ret),title('�渵��Ҷ�任');