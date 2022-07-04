
%%%%%%%%%%%%%%%��˹��ͨ�˲�%%%%%%%%%%%%%%%%


clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
% RGB=imread('timg.jpg');
RGB=imread([pn fn ]);
%RGB=im2double(RGB); %���п���
%%%%%%image=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������

R=RGB(:,:,1);%ΪR��������
G=(RGB(:,:,2));%ΪG��������
B=(RGB(:,:,3));%ΪB��������



tic
YUV=rgb2ycbcr(RGB);%ת��YUV
toc
%%%����ͨ��%%%%%%%%%%
Y=(YUV(:,:,1));%ΪY��������
U=(YUV(:,:,2));%ΪU��������
V=(YUV(:,:,3));%ΪV��������


%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv = cat(3, Y, U, V);  

%%%%%��ʾͼ��İ˹���%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCrͼ��Yͨ����Cbͨ����Crͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('YCbCr');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
 
 %Y=im2double(Y);
  F=fft2(Y);          %����Ҷ�任
 F1=log(abs(F)+1);   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ �鲿ȥ��������
  %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
  S=log(abs(Fs)+1);    %ȡģ����������
FFt= fftshift(F1);   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼNO real
  Y=S;
% % I_C_rep=im2uint8(real(ifft2(J)));
  yuv= cat(3, Y,U,V); 

 
  tic
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
           % D = sqrt((i-width/2)^2+(j-high/2)^2);%%��˹
        %%%%%%%%%%%%������˹��ͨ�˲���%%%%%%%%% 
        h0 = exp(-1/2*(d.^2)/(D0*D0));%%%%%��˹
        %h0=1/(1+0.414*(d./D0)^(2*n4)); %����D0=10;��ͨ�˲������ݺ���
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;�˲����������ͼ��
               T0(i, j) = h0;                           %%D0=10;�˲���������
          
                h1= exp(-1/2*(d.^2)/(D1*D1));%%%%%��˹
        %h1=1/(1+0.414*(d./D1)^(2*n4));%�����ͨ�˲������ݺ���
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;�˲����������ͼ��
              T1(i, j) = h1;                          %%%%D1=20;�˲���������
       
               h2 = exp(-1/2*(d.^2)/(D2*D2));%%%%%��˹
        %h2=1/(1+0.414*(d./D2)^(2*n4));%����D2=40;��ͨ�˲������ݺ���
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;�˲����������ͼ��
               T2(i, j) = h2;                         %%%%D2=40;�˲���������
               
                h3 = exp(-1/2*(d.^2)/(D3*D3));%%%%%��˹
       % h3=1/(1+0.414*(d./D3)^(2*n4)); %����D3=60;��ͨ�˲������ݺ���
                s3(i,j)=h3*Fs(i,j);                  %D3=60;�˲����������ͼ��
                T3(i, j) = h3;                         %%;D3=60;�˲���������
        
                 h4 = exp(-1/2*(d.^2)/(D4*D4));%%%%%��˹
        %h4=1/(1+0.414*(d./D4)^(2*n4)); %����D4=80;��ͨ�˲������ݺ���
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;�˲����������ͼ��
                T4(i, j) = h4;                          %%D4=80;�˲���������
                 
                h5 = exp(-1/2*(d.^2)/(D5*D5));%%%%%��˹
        %h5=1/(1+0.414*(d./D5)^(2*n4)); %����D5=255;��ͨ�˲������ݺ���
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
toc
figure('NumberTitle', 'off', 'Name', 'Yͨ��Ƶ�׸�˹�˲���ʾͼ');
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
subplot(3,4,11);imshow(fr54,[]);title('D5=255 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');
% % 
% % 
% % 
%%%%%%%%%yuv lowpass filter result%%%%%%%%%%%
% % % % % % %    rgb = cat(3, R, G, B);  
yuv0 = cat(3, fr0, U, V);  
yuv1= cat(3, fr10, U, V);  
yuv2 = cat(3, fr21, U, V);  
yuv3 = cat(3, fr32, U, V);  
yuv4 = cat(3,  fr43,U, V);  
yuv5 = cat(3, fr54,U,V);  

yuv0p= cat(3, fr0,U,V);  
yuv1p= cat(3, fr1, U, V);  
yuv2p= cat(3, fr2, U, V);  
yuv3p = cat(3, fr3, U, V);  
yuv4p= cat(3,  fr4,U, V);  
yuv5p= cat(3, fr5,U,V);  
figure('NumberTitle', 'off', 'Name', '=>YCbCr012345,YCbCr0p1p2p3p4p5p');
subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(yuv0p,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(yuv1p,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(yuv2p);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(yuv3p,[]),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(yuv4p),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(yuv5p,[]),title('��ͨ�˲���D5=255 D5-D4');

% % figure('NumberTitle', 'off', 'Name', '5=>YUV6����ͬƵ���˲������Ӽ������ͼ��');
% % subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
% % subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% % subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% % subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% % subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% % subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% % subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% % subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% % subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% % subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% % subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% % subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
% % 
% % %  figure('NumberTitle', 'off', 'Name', '55=>YUV6����ͬƵ���˲������Ӽ������ͼ��');
% % % subplot(3,4,1);imshow(yuv0P,[]);title('D0=10��Ч��ͼ');
% % % subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% % % subplot(3,4,3);imshow(yuv1P,[]);title('D1=20 D1-D0');
% % % subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% % % subplot(3,4,5);imshow(yuv2P,[]);title('D2=40 D2-D1');
% % % subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% % % subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% % % subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% % % subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% % % subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% % % subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% % % subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');

 RGB_0= ycbcr2rgb(yuv0);%ת��RGB
 RGB_1= ycbcr2rgb(yuv1);
 RGB_2= ycbcr2rgb(yuv2);
 RGB_3= ycbcr2rgb(yuv3);
 RGB_4= ycbcr2rgb(yuv4);
  RGB_5= ycbcr2rgb(yuv5);
  rgb0= ycbcr2rgb(yuv0p);%ת��RGB
 rgb1= ycbcr2rgb(yuv1p);
 rgb2= ycbcr2rgb(yuv2p);
 rgb3= ycbcr2rgb(yuv3p);
 rgb4= ycbcr2rgb(yuv4p);
  rgb5= ycbcr2rgb(yuv5p);
  figure('NumberTitle', 'off', 'Name', '=>RGB012345,RGB0p1p2p3p4p5p');
subplot(3,4,1);imshow(RGB_0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(rgb0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(RGB_1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(rgb1,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(RGB_2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(rgb2);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(RGB_3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(rgb3,[]),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(RGB_4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(rgb4 ),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(RGB_5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(rgb5,[]),title('��ͨ�˲���D5=255 D5-D4');

%     K5 = imlincomb(0.5,k4,0.5,RGB_5);
%  RGB_0=imlincomb(1,RGB_0);%I,J�Ƕ��������ͼ��
%  RGB_01=imlincomb(1,RGB_0,0.9,RGB_1);
%    RGB_012=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2);
% %   RGB_012=imadd(0.7*RGB_012,0.7*RGB_3);
%    RGB_0123=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3);
% %    RGB_01234=imadd(0.3*RGB_0123,0.7*RGB_4);
% RGB_01234=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3,0.6,RGB_4);
%       RGB_012345=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3,0.6,RGB_4,0.5,RGB_5);
%     

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(0.9*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
       S=abs( RGB_012345)+1;    %ȡģ����������
 figure('NumberTitle', 'off', 'Name', '6=>YCbCr��ԭRGB��ͬ6Ƶ���˲������Ӽ������ͼ��');
 

 subplot(231),imshow(RGB_0,[])  ,title('YCbCr-0Ƶ�θ���');
 subplot(232),imshow(RGB_01,[])  ,title('YCbCr-01Ƶ�θ���');
 subplot(233),imshow( RGB_012,[])  ,title('YCbCr-012Ƶ�θ���');
 subplot(234),imshow( RGB_0123,[])  ,title('YCbCr-0123Ƶ�θ���');
subplot(235),imshow( RGB_01234,[])  ,title('YCbCr-01234Ƶ�θ���');
 subplot(236),imshow(RGB_012345,[])  ,title('YCbCr-012345Ƶ�θ���');








%%%%%%%%%%%%%%%��˹��ͨ�˲�%%%%%%%%%%%%%%%%


clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
 [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
% RGB=imread('timg.jpg');
RGB=imread([pn fn ]);
%RGB=im2double(RGB); %���п���
%%%%%%image=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������

R=RGB(:,:,1);%ΪR��������
G=(RGB(:,:,2));%ΪG��������
B=(RGB(:,:,3));%ΪB��������



tic
YUV=rgb2ycbcr(RGB);%ת��YUV
toc
%%%����ͨ��%%%%%%%%%%
Y=(YUV(:,:,1));%ΪY��������
U=(YUV(:,:,2));%ΪU��������
V=(YUV(:,:,3));%ΪV��������


%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv = cat(3, Y, U, V);  

%%%%%��ʾͼ��İ˹���%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'YCbCrͼ��Yͨ����Cbͨ����Crͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('YCbCr');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
 
 %Y=im2double(Y);
  F=fft2(Y);          %����Ҷ�任
 F1=log(abs(F)+1);   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ �鲿ȥ��������
  %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
  S=log(abs(Fs)+1);    %ȡģ����������
FFt= fftshift(F1);   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼNO real
  Y=S;
% % I_C_rep=im2uint8(real(ifft2(J)));
  yuv= cat(3, Y,U,V); 

 
  tic
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
           % D = sqrt((i-width/2)^2+(j-high/2)^2);%%��˹
        %%%%%%%%%%%%������˹��ͨ�˲���%%%%%%%%% 
        h0 = exp(-1/2*(d.^2)/(D0*D0));%%%%%��˹
        %h0=1/(1+0.414*(d./D0)^(2*n4)); %����D0=10;��ͨ�˲������ݺ���
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;�˲����������ͼ��
               T0(i, j) = h0;                           %%D0=10;�˲���������
          
                h1= exp(-1/2*(d.^2)/(D1*D1));%%%%%��˹
        %h1=1/(1+0.414*(d./D1)^(2*n4));%�����ͨ�˲������ݺ���
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;�˲����������ͼ��
              T1(i, j) = h1;                          %%%%D1=20;�˲���������
       
               h2 = exp(-1/2*(d.^2)/(D2*D2));%%%%%��˹
        %h2=1/(1+0.414*(d./D2)^(2*n4));%����D2=40;��ͨ�˲������ݺ���
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;�˲����������ͼ��
               T2(i, j) = h2;                         %%%%D2=40;�˲���������
               
                h3 = exp(-1/2*(d.^2)/(D3*D3));%%%%%��˹
       % h3=1/(1+0.414*(d./D3)^(2*n4)); %����D3=60;��ͨ�˲������ݺ���
                s3(i,j)=h3*Fs(i,j);                  %D3=60;�˲����������ͼ��
                T3(i, j) = h3;                         %%;D3=60;�˲���������
        
                 h4 = exp(-1/2*(d.^2)/(D4*D4));%%%%%��˹
        %h4=1/(1+0.414*(d./D4)^(2*n4)); %����D4=80;��ͨ�˲������ݺ���
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;�˲����������ͼ��
                T4(i, j) = h4;                          %%D4=80;�˲���������
                 
                h5 = exp(-1/2*(d.^2)/(D5*D5));%%%%%��˹
        %h5=1/(1+0.414*(d./D5)^(2*n4)); %����D5=255;��ͨ�˲������ݺ���
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
toc
figure('NumberTitle', 'off', 'Name', 'Yͨ��Ƶ�׸�˹�˲���ʾͼ');
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
subplot(3,4,11);imshow(fr54,[]);title('D5=255 D5-D4');
subplot(3,4,12);imshow(T5-T4),title('��ͨ�˲���D5=255 D5-D4');
% % 
% % 
% % 
%%%%%%%%%yuv lowpass filter result%%%%%%%%%%%
% % % % % % %    rgb = cat(3, R, G, B);  
yuv0 = cat(3, fr0, U, V);  
yuv1= cat(3, fr10, U, V);  
yuv2 = cat(3, fr21, U, V);  
yuv3 = cat(3, fr32, U, V);  
yuv4 = cat(3,  fr43,U, V);  
yuv5 = cat(3, fr54,U,V);  

yuv0p= cat(3, fr0,U,V);  
yuv1p= cat(3, fr1, U, V);  
yuv2p= cat(3, fr2, U, V);  
yuv3p = cat(3, fr3, U, V);  
yuv4p= cat(3,  fr4,U, V);  
yuv5p= cat(3, fr5,U,V);  
figure('NumberTitle', 'off', 'Name', '=>YCbCr012345,YCbCr0p1p2p3p4p5p');
subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(yuv0p,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(yuv1p,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(yuv2p);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(yuv3p,[]),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(yuv4p),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(yuv5p,[]),title('��ͨ�˲���D5=255 D5-D4');

% % figure('NumberTitle', 'off', 'Name', '5=>YUV6����ͬƵ���˲������Ӽ������ͼ��');
% % subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
% % subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% % subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% % subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% % subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% % subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% % subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% % subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% % subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% % subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% % subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% % subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
% % 
% % %  figure('NumberTitle', 'off', 'Name', '55=>YUV6����ͬƵ���˲������Ӽ������ͼ��');
% % % subplot(3,4,1);imshow(yuv0P,[]);title('D0=10��Ч��ͼ');
% % % subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% % % subplot(3,4,3);imshow(yuv1P,[]);title('D1=20 D1-D0');
% % % subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% % % subplot(3,4,5);imshow(yuv2P,[]);title('D2=40 D2-D1');
% % % subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% % % subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% % % subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% % % subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% % % subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% % % subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% % % subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');

 RGB_0= ycbcr2rgb(yuv0);%ת��RGB
 RGB_1= ycbcr2rgb(yuv1);
 RGB_2= ycbcr2rgb(yuv2);
 RGB_3= ycbcr2rgb(yuv3);
 RGB_4= ycbcr2rgb(yuv4);
  RGB_5= ycbcr2rgb(yuv5);
  rgb0= ycbcr2rgb(yuv0p);%ת��RGB
 rgb1= ycbcr2rgb(yuv1p);
 rgb2= ycbcr2rgb(yuv2p);
 rgb3= ycbcr2rgb(yuv3p);
 rgb4= ycbcr2rgb(yuv4p);
  rgb5= ycbcr2rgb(yuv5p);
  figure('NumberTitle', 'off', 'Name', '=>RGB012345,RGB0p1p2p3p4p5p');
subplot(3,4,1);imshow(RGB_0,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(rgb0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(RGB_1,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(rgb1,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(RGB_2,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(rgb2);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(RGB_3,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(rgb3,[]),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(RGB_4,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(rgb4 ),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(RGB_5,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(rgb5,[]),title('��ͨ�˲���D5=255 D5-D4');

%     K5 = imlincomb(0.5,k4,0.5,RGB_5);
%  RGB_0=imlincomb(1,RGB_0);%I,J�Ƕ��������ͼ��
%  RGB_01=imlincomb(1,RGB_0,0.9,RGB_1);
%    RGB_012=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2);
% %   RGB_012=imadd(0.7*RGB_012,0.7*RGB_3);
%    RGB_0123=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3);
% %    RGB_01234=imadd(0.3*RGB_0123,0.7*RGB_4);
% RGB_01234=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3,0.6,RGB_4);
%       RGB_012345=imlincomb(1,RGB_0, 0.9,RGB_1, 0.8,RGB_2,0.7,RGB_3,0.6,RGB_4,0.5,RGB_5);
%     

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(0.9*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
       S=abs( RGB_012345)+1;    %ȡģ����������
 figure('NumberTitle', 'off', 'Name', '6=>YCbCr��ԭRGB��ͬ6Ƶ���˲������Ӽ������ͼ��');
 

 subplot(231),imshow(RGB_0,[])  ,title('YCbCr-0Ƶ�θ���');
 subplot(232),imshow(RGB_01,[])  ,title('YCbCr-01Ƶ�θ���');
 subplot(233),imshow( RGB_012,[])  ,title('YCbCr-012Ƶ�θ���');
 subplot(234),imshow( RGB_0123,[])  ,title('YCbCr-0123Ƶ�θ���');
subplot(235),imshow( RGB_01234,[])  ,title('YCbCr-01234Ƶ�θ���');
 subplot(236),imshow(RGB_012345,[])  ,title('YCbCr-012345Ƶ�θ���');













