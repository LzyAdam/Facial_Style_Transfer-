clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
% [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������
clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
 tic
 RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', 'ԭͼ');%%figure������


tic
YUV=rgb2ycbcr(RGB);%ת��YUV
toc                                                              % % figure
                                                                            % % imshow(YUV);

%%%����ͨ��%%%%%%%%%%
Y=YUV(:,:,1);%ΪY��������
U=YUV(:,:,2);%ΪU��������
V=YUV(:,:,3);%ΪV��������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv = cat(3, Y, U, V);  
%%%%%%��ʾͼ��İ˹���%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCrͼ��Yͨ����Cbͨ����Crͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('YCbCr');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
 %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%���½��и���Ҷ�任%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(Y);          %����Ҷ�任
%   F1=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
  Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ��Ҳ�Ծ���ֵ���ı���
  %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
                                        %%ע��Fs=F1,���fs=f,�ͳ�������
                                        %%%����Fs=F����6���˲����˳���ͼ�ɼ�����Fs=F1���ɼ�                         
%   S=log(abs(Fs)+1);    %ȡģ����������
%   FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
%   Y=FFt;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%��ԭ����YUV2RGB%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y=ifft2(ifftshift(Fs));
 yuv= cat(3, Y,U,V); 
 RGB_ = ycbcr2rgb(YUV);%ת��RGB
 figure('NumberTitle', 'off', 'Name', 'YCbCrͼ��2RGB_');
 imshow(RGB_);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%�����Կ���������ԭRGBͼ��%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%dispaly the fft result%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCr��Yͨ���ĸ���ҶƵ��ͼ��YCbCrͼ��RGBͼ');
subplot(131),imshow(Y,[])  ,title('Yͨ��Ƶ��ͼ');
subplot(132),imshow(yuv,[])  ,title('YCbCr=>FFT��ͨ��ͼ');
subplot(133),imshow(RGB_,[])  ,title('RGBͼ');
%%%%%%%dispaly the fft result%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%��������ͨ�˲���������˹�˲�%%%%%%%%%%
 %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
 
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
figure('NumberTitle', 'off', 'Name', '6����ͬƵ���˲������Ӽ������ͼ��');
subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');%%D0=10;�˲����������ͼ��
subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');%%D0=10;�˲�������
subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;�˲����������ͼ��
subplot(3,4,4);imshow(T1-T0),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');

yuv0 = cat(3, fr0,U,V);  
yuv1= cat(3, fr10, U, V);  
yuv2 = cat(3, fr21, U, V);  
yuv3 = cat(3, fr32, U, V);  
yuv4 = cat(3,  fr43,U, V);  
yuv5 = cat(3, fr54,U,V);  
RGB_0= ycbcr2rgb(yuv0);%ת��RGB
 RGB_1= ycbcr2rgb(yuv1);
 RGB_2= ycbcr2rgb(yuv2);
 RGB_3= ycbcr2rgb(yuv3);
 RGB_4= ycbcr2rgb(yuv4);
  RGB_5= ycbcr2rgb(yuv5);
figure('NumberTitle', 'off', 'Name', 'YCbCr����6����ͬƵ���˲������Ӽ������ͼ��');
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

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(1*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
       S=abs( RGB_012345)+1;    %ȡģ����������
 figure('NumberTitle', 'off', 'Name', '6=>YCbCr��ԭRGB��ͼ��');
 


imshow(RGB_012345,[])  ,title('6Ƶ�θ���');






