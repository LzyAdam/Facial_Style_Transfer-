clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
 clc
% [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������
RGB=imread('timg.jpg');
%  RGB=im2double(RGB);%%��Ӱ��
R= RGB(:, :, 1); 
G= RGB(:, :, 2); 
B= RGB(:, :, 3); 
% % % % % % % tic
% % % % % % % HSV=rgb2hsv(RGB);%ת��HSV
% % % % % % %  toc                                                              % % figure
% Y=zeros(m,n);   %����%RGB2YUV
% U=zeros(m,n);   %�ʶ�
% V=zeros(m,n);   %Ũ��
Y=0.299.*R+0.578.*G+0.114.*B;
U=-0.1687.*R-0.3313.*G+0.5.*B+128;
V=0.5.*R-0.4187.*G-0.0813.*B+128;
yuv = cat(3, Y, U, V);  
toc
figure('NumberTitle', 'off', 'Name', 'YUVͼ��');
imshow(yuv,[]);title('rgb2yuv');
% tic
% YUV=rgb2ycbcr(RGB);%ת��YUV
% toc                                                              % % figure
                                                                            % % imshow(YUV);

%%%����ͨ��%%%%%%%%%%
Y=yuv(:,:,1);%ΪY��������
U=yuv(:,:,2);%ΪU��������
V=yuv(:,:,3);%ΪV��������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv = cat(3, Y, U, V);                                                                       % % imshow(YUV);


%%%����ͨ��%%%%%%%%%%
% % % % % H=HSV(:,:,1);%ΪY��������* 2 * pi
% % % % % S=HSV(:,:,2);%ΪU��������
% % % % % V=HSV(:,:,3);%ΪV��������
% % % % % RGB = hsv2rgb(HSV) ;
% imshow(RGB);%%%figure1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
% % % % hsv = cat(3, H, S, V);  
%%%%%%��ʾͼ��İ˹���%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '1=>HSVͼ��Hͨ����Sͨ����Vͨ�� �ֱ���ʾ');%%figure������
% subplot(221),imshow(hsv),title('HSV');
% subplot(222),imshow(H,[]),title('H');
%  subplot(223),imshow(S),title('S');
%  subplot(224),imshow(V),title('V');
%  %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%���½��и���Ҷ�任%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(Y);          %����Ҷ�任
  F1=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
  Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ��Ҳ�Ծ���ֵ���ı���
  %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
                                        %%ע��Fs=F1,���fs=f,�ͳ�������
                                        %%%����Fs=F����6���˲����˳���ͼ�ɼ�����Fs=F1���ɼ�                         
%   S=log(abs(Fs)+1);    %ȡģ����������
%   FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
%  V=FFt;�����Щ����ֱ��Ū
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%��ԭ����YUV2RGB%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V=ifft2(ifftshift(Fs));
%  hsv= cat(3, H ,S , V); 
%  RGB_ = hsv2rgb(hsv);%ת��RGB
%  figure('NumberTitle', 'off', 'Name', '2=>YUVͼ��2RGB_');
%  imshow(RGB_);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%�����Կ���������ԭRGBͼ��%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%dispaly the fft result%%%%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '3=>HSV��Vͨ���ĸ���ҶƵ��ͼ��HSVͼ��RGBͼ');
% subplot(131),imshow(V,[])  ,title('Vͨ��Ƶ��ͼ');
% subplot(132),imshow(hsv,[])  ,title('HSV=>FFT��ͨ��ͼ');
% subplot(133),imshow(RGB_,[])  ,title('RGBͼ');
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
        h0=1; %����D0=10;��ͨ�˲������ݺ���
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;�˲�����������ͼ��
               T0(i, j) = h0;                           %%D0=10;�˲���������
               
        h1=1;%�����ͨ�˲������ݺ���
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;�˲�����������ͼ��
              T1(i, j) = h1;                          %%%%D1=20;�˲���������
       
        h2=1;%����D2=40;��ͨ�˲������ݺ���
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;�˲�����������ͼ��
               T2(i, j) = h2;                         %%%%D2=40;�˲���������
               
        h3=1; %����D3=60;��ͨ�˲������ݺ���
                s3(i,j)=h3*Fs(i,j);                  %D3=60;�˲�����������ͼ��
                T3(i, j) = h3;                         %%;D3=60;�˲���������
                
        h4=1; %����D4=80;��ͨ�˲������ݺ���
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;�˲�����������ͼ��
                T4(i, j) = h4;                          %%D4=80;�˲���������
                
        h5=1; %����D5=255;��ͨ�˲������ݺ���
                s5(i,j)=h5*Fs(i,j);                  %%D5=255;�˲�����������ͼ��
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
% figure('NumberTitle', 'off', 'Name', '4=>6����ͬƵ���˲������Ӽ�������ͼ��');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');%%D0=10;�˲�����������ͼ��
% subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');%%D0=10;�˲�������
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;�˲�����������ͼ��
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
% % % % hsv0 = cat(3,H,S,fr0);  
% % % % hsv1= cat(3, H,S,fr10);  
% % % % hsv2 = cat(3, H,S,fr21);  
% % % % hsv3 = cat(3, H,S,fr32);  
% % % % hsv4 = cat(3,  H,S,fr43);  
% % % % hsv5 = cat(3, H,S,fr54);  
% % % % RGB_0= hsv2rgb(hsv0);%ת��RGB
% % % %  RGB_1= hsv2rgb(hsv1);
% % % %  RGB_2= hsv2rgb(hsv2);
% % % %  RGB_3= hsv2rgb(hsv3);
% % % %  RGB_4= hsv2rgb(hsv4);
% % % %   RGB_5= hsv2rgb(hsv5);
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
  figure('NumberTitle', 'off', 'Name', 'YUV����6����ͬƵ���˲������Ӽ�������ͼ��');
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
 figure('NumberTitle', 'off', 'Name', '6=>YUV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
  
imshow(RGB_012345,[])  ,title('6Ƶ�θ���');

% figure('NumberTitle', 'off', 'Name', '5=>HSV����6����ͬƵ���˲������Ӽ�������ͼ��');
% subplot(3,4,1);imshow(hsv0,[]);title('D0=10��Ч��ͼ');
% subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% subplot(3,4,3);imshow(hsv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% subplot(3,4,5);imshow(hsv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% subplot(3,4,7);imshow(hsv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% subplot(3,4,9);imshow(hsv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% subplot(3,4,11);imshow(hsv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(0.9*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
%        S=abs( RGB_012345)+1;    %ȡģ����������
%  figure('NumberTitle', 'off', 'Name', '6=>HSV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
%   subplot(231),imshow(RGB_0,[])  ,title('HSV0Ƶ�θ���');
%  subplot(232),imshow(RGB_01,[])  ,title('HSV01Ƶ�θ���');
%  subplot(233),imshow( RGB_012,[])  ,title('HSV012Ƶ�θ���');
%  subplot(234),imshow( RGB_0123,[])  ,title('HSV0123Ƶ�θ���');
% subplot(235),imshow( RGB_01234,[])  ,title('HSV01234Ƶ�θ���');
%  subplot(236),imshow(RGB_012345,[])  ,title('HSV012345Ƶ�θ���');
 P0=sumsqr(F1);
%  P0=sumsqr(RGB);%%%%RGB ��ԭʼ��ͼ��
 A0= log(abs(s0)+1);A1= log(abs(s1)+1);A2= log(abs(s2)+1);
 A3= log(abs(s3)+1);A4= log(abs(s4)+1);A5= log(abs(s5)+1);
 P0=sumsqr(F1);%%%%RGB ��ԭʼ��ͼ��
P1=sumsqr(A0)/P0;P2=sumsqr(A1)/P0;P3=sumsqr(A2)/P0;
P4=sumsqr(A3)/P0;P5=sumsqr(A4)/P0;P6=sumsqr(A5)/P0;
 





      

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%�ڶ���ͼ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % % % % % % % [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%���￪ʼ%%%%%%%%%%%%%%%%%%%
RGB2=imread('altlas11.jpg');
%  RGB2=im2double(RGB2);
 
R2= RGB2(:, :, 1); 
G2= RGB2(:, :, 2); 
B2= RGB2(:, :, 3); 
% R= RGB(:, :, 1); 
% G= RGB(:, :, 2); 
% B= RGB(:, :, 3); 
% % % % % % % tic
% % % % % % % HSV=rgb2hsv(RGB);%ת��HSV
% % % % % % %  toc                                                              % % figure
% Y=zeros(m,n);   %����%RGB2YUV
% U=zeros(m,n);   %�ʶ�
% V=zeros(m,n);   %Ũ��
Y2=0.299.*R2+0.578.*G2+0.114.*B2;
U2=-0.1687.*R2-0.3313.*G2+0.5.*B2+128;
V2=0.5.*R2-0.4187.*G2-0.0813.*B2+128;
yuv2= cat(3, Y2, U2, V2);  
toc
figure('NumberTitle', 'off', 'Name', 'YUV2222222ͼ��');
imshow(yuv2,[]);title('rgb2yuv');
% tic
% YUV=rgb2ycbcr(RGB);%ת��YUV
% toc                                                              % % figure
                                                                            % % imshow(YUV);

%%%����ͨ��%%%%%%%%%%
Y2=yuv2(:,:,1);%ΪY��������
U2=yuv2(:,:,2);%ΪU��������
V2=yuv2(:,:,3);%ΪV��������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
yuv2 = cat(3, Y2, U2, V2);         
% tic
% HSV2=rgb2hsv(RGB2);%ת��HSV
%  toc                                                             
%  figure
%                                                                            imshow(HSV2);
% 
% 
% %%%����ͨ��%%%%%%%%%%
% H2=HSV2(:,:,1);%ΪY��������* 2 * pi
% S2=HSV2(:,:,2);%ΪU��������
% V2=HSV2(:,:,3);%ΪV��������
% RGB2 = hsv2rgb(HSV2) ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
% hsv2 = cat(3, H2, S2, V2);  
% %%%%%%��ʾͼ��İ˹���%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '7=>HSVͼ��Hͨ����Sͨ����Vͨ�� �ֱ���ʾ');%%figure������
% subplot(221),imshow(HSV2),title('HSV2');
% subplot(222),imshow(H2,[]),title('H2');
%  subplot(223),imshow(S2),title('S2');
%  subplot(224),imshow(V2),title('V2');
%  %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%���½��и���Ҷ�任%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F2=fft2(Y2);          %����Ҷ�任
   F1t=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
  Fs2=fftshift(F2);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ��Ҳ�Ծ���ֵ���ı���
  %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
%                                         %%ע��Fs=F1,���fs=f,�ͳ�������
%                                         %%%����Fs=F����6���˲����˳���ͼ�ɼ�����Fs=F1���ɼ�                         
% %   S=log(abs(Fs)+1);    %ȡģ����������
% %   FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
% %  V=FFt;�����Щ����ֱ��Ū
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%��ԭ����YUV2RGB%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2=ifft2(ifftshift(Fs2));
% yuv2 = cat(3, Y2, U2, V2);         
%  RGB2= hsv2rgb(hsv2);%ת��RGB
%  figure('NumberTitle', 'off', 'Name', '8=>HSVͼ��2RGB_');
%  imshow(RGB2);
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%�����Կ���������ԭRGBͼ��%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%dispaly the fft result%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', '99999=>HSV��Vͨ���ĸ���ҶƵ��ͼ��HSVͼ��RGBͼ');
subplot(131),imshow(Y2,[])  ,title('Y2ͨ��Ƶ��ͼ');
subplot(132),imshow(yuv2,[])  ,title('HSV=>FFT��ͨ��ͼ');
subplot(133),imshow(RGB2,[])  ,title('RGBͼ');
% % %%%%%%%dispaly the fft result%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%��������ͨ�˲���������˹�˲�%%%%%%%%%%
% %  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
% %  
 n4=2;%step1%�˲����Ľ���2%%%%%%
 %%step2%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
%%%step3%%%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
 [M2,N2]=size(F2);%%%%%�˲�����С������ͼƬ%%%%%%%
m2=fix(M2/2);
n2=fix(N2/2);
for i=1:M2
   for j=1:N2
        d=sqrt((i-m2)^2+(j-n2)^2);%%%%��㵽ͼ�����ľ���%%%%%%%  
        
        %%%%%%%%%%%%������˹��ͨ�˲���%%%%%%%%% 
        h6=1; %����D0=10;��ͨ�˲������ݺ���
               s6(i,j)=h6*Fs2(i,j);                   %%D0=10;�˲�����������ͼ��
               T6(i, j) = h6;                           %%D0=10;�˲���������
               
        h7=1;%�����ͨ�˲������ݺ���
              s7(i,j)=h7*Fs2(i,j);                   %%%%D1=20;�˲�����������ͼ��
              T7(i, j) = h7;                          %%%%D1=20;�˲���������
       
        h8=1;%����D2=40;��ͨ�˲������ݺ���
               s8(i,j)=h8*Fs2(i,j);                  %%D2=40;�˲�����������ͼ��
               T8(i, j) = h8;                         %%%%D2=40;�˲���������
               
        h9=1; %����D3=60;��ͨ�˲������ݺ���
                s9(i,j)=h9*Fs2(i,j);                  %D3=60;�˲�����������ͼ��
                T9(i, j) = h9;                         %%;D3=60;�˲���������
                
        h10=1; %����D4=80;��ͨ�˲������ݺ���
                s10(i,j)=h10*Fs2(i,j);                  %%D4=80;�˲�����������ͼ��
                T10(i, j) = h10;                          %%D4=80;�˲���������
                
        h11=1; %����D5=255;��ͨ�˲������ݺ���
                s11(i,j)=h11*Fs2(i,j);                  %%D5=255;�˲�����������ͼ��
                T11(i, j) = h11;                         %%D5=255;�˲���������
       
   end
end
fr6=real(ifft2(ifftshift(s6)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
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
figure('NumberTitle', 'off', 'Name', '1000=>226����ͬƵ���˲������Ӽ�������ͼ��');
subplot(3,4,1);imshow(fr6,[]);title('D0=10��Ч��ͼ');%%D0=10;�˲�����������ͼ��
subplot(3,4,2);imshow(T6);title('��ͨ�˲���D0=10');%%D0=10;�˲�������
subplot(3,4,3);imshow(fr76,[]);title('D1=20 D1-D0');%%D1-D0;�˲�����������ͼ��
subplot(3,4,4);imshow(T7-T6),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(fr87,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T8-T7);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(fr98,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T9-T8),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(fr109,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T10-T9),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(fr1110,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T11-T10),title('��ͨ�˲���D5=255 D5-D4');

% hsv6 = cat(3,H2,S2,fr6);  
% hsv7= cat(3, H2,S2,fr76);  
% hsv8 = cat(3, H2,S2,fr87);  
% hsv9 = cat(3, H2,S2,fr98);  
% hsv10 = cat(3,  H2,S2,fr109);  
% hsv11= cat(3, H2,S2,fr1110);  
% RGB_6= hsv2rgb(hsv6);%ת��RGB
%  RGB_7= hsv2rgb(hsv7);
%  RGB_8= hsv2rgb(hsv8);
%  RGB_9= hsv2rgb(hsv9);
%  RGB_10= hsv2rgb(hsv10);
%   RGB_11= hsv2rgb(hsv11);
yuv6 = cat(3, fr6,U2,V2);  
yuv7= cat(3, fr76, U2, V2);  
yuv8 = cat(3, fr87, U2, V2);  
yuv9 = cat(3, fr98, U2, V2);  
yuv10 = cat(3,  fr109,U2, V2);  
yuv11= cat(3, fr1110,U2,V2);  
RGB_6= ycbcr2rgb(yuv6);%ת��RGB
 RGB_7= ycbcr2rgb(yuv7);
 RGB_8= ycbcr2rgb(yuv8);
 RGB_9= ycbcr2rgb(yuv9);
 RGB_10= ycbcr2rgb(yuv10);
  RGB_11= ycbcr2rgb(yuv11);
  figure('NumberTitle', 'off', 'Name', 'YUV����6����ͬƵ���˲������Ӽ�������ͼ��');
subplot(3,4,1);imshow(yuv6,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(yuv7,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(yuv8,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(yuv9,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(yuv10,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv11,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
% % RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
% %  RGB_012=imadd(1*RGB_01,0.8*RGB_2);
% %   RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
% %    RGB_01234=imadd(1*RGB_0123,0.6*RGB_4);
% %       RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
% %        S=abs( RGB_012345)+1;    %ȡģ����������
 
% %   figure('NumberTitle', 'off', 'Name', 'YUV����6����ͬƵ���˲������Ӽ�������ͼ��');
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
figure('NumberTitle', 'off', 'Name', '11=>����6����ͬƵ���˲������Ӽ�������ͼ��');
subplot(3,4,1);imshow(yuv6,[]);title('D0=10��Ч��ͼ');
subplot(3,4,2);imshow(T6,[]);title('��ͨ�˲���D0=10');
subplot(3,4,3);imshow(yuv7,[]);title('D1=20 D1-D0');
subplot(3,4,4);imshow(T7-T6,[]),title('��ͨ�˲���D1=20 D1-D0');
subplot(3,4,5);imshow(yuv8,[]);title('D2=40 D2-D1');
subplot(3,4,6);imshow(T8-T7);title('��ͨ�˲���D2=40 D2-D1');
subplot(3,4,7);imshow(yuv9,[]);title('D3=60 D3-D2');
subplot(3,4,8);imshow(T9-T8),title('��ͨ�˲���D3=60 D3-D2');
subplot(3,4,9);imshow(yuv10,[]);title('D4=80 D4-D3');
subplot(3,4,10);imshow(T10-T9),title('��ͨ�˲���D4=60 D4-D3 ');
subplot(3,4,11);imshow(yuv11,[]);title('D5=80 D5-D4');
subplot(3,4,12);imshow(T11-T10,[]),title('��ͨ�˲���D5=255 D5-D4');


RGB_67=imadd(1*RGB_6,0.9*RGB_7);%I,J�Ƕ��������ͼ��
 RGB_678=imadd(1*RGB_67,0.8*RGB_8);
  RGB_6789=imadd(1*RGB_678,0.7*RGB_9);
   RGB_678910=imadd(0.9*RGB_6789,0.6*RGB_10);
      RGB_67891011=imadd(1.2*RGB_678910,1*RGB_11);
  figure('NumberTitle', 'off', 'Name', '6=>YUV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
  
imshow(RGB_67891011,[])  ,title('6Ƶ�θ���');
% % 
%  figure('NumberTitle', 'off', 'Name', '12=>=>HSV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
%  subplot(231),imshow(RGB_6,[])  ,title('HSV6Ƶ�θ���');
%  subplot(232),imshow(RGB_67,[])  ,title('HSV67Ƶ�θ���');
%  subplot(233),imshow( RGB_678,[])  ,title('HSV678Ƶ�θ���');
%  subplot(234),imshow( RGB_6789,[])  ,title('HSV6789Ƶ�θ���');
% subplot(235),imshow( RGB_678910,[])  ,title('HSV678910Ƶ�θ���');
%  subplot(236),imshow(RGB_67891011,[])  ,title('HSV67891011Ƶ�θ���');
%  
 Pt=sumsqr(F1t);
%  P0=sumsqr(RGB);%%%%RGB ��ԭʼ��ͼ��
 A6= log(abs(s6)+1);A7= log(abs(s7)+1);A8= log(abs(s8)+1);
 A9= log(abs(s9)+1);A10= log(abs(s10)+1);A11= log(abs(s11)+1);
% % %  ���������ٷֱ�
P7=sumsqr(A6)/Pt;P8=sumsqr(A7)/Pt;P9=sumsqr(A8)/Pt;
P10=sumsqr(A9)/Pt;P11=sumsqr(A10)/Pt;P12=sumsqr(A11)/Pt;
%�������ļ���
% G1=sqrt(P1/((P7+P1)+5));G2=sqrt(P2/((P8+P2)*10));G3=sqrt(P3/((P9+P3)*10));
%   G4=sqrt(P4/((P10+P4)*10));G5=sqrt(P5/((P11+P5)*10));G6=sqrt(P6/((P12+P6)*10));
%  G1=sqrt(P1/((P7+P1)*10));G2=sqrt(P2/((P8+P2)*10));G3=sqrt(P3/((P9+P3)*10));
%   G4=sqrt(P4/((P10+P4)*10));G5=sqrt(P5/((P11+P5)*10));G6=sqrt(P6/((P12+P6)*10));
  G1=sqrt((P1)/((P7+0.1)));G2=sqrt((P2)/((P8+0.1)));G3=sqrt((P3)/((P9+0.1)));
  G4=sqrt((P4)/((P10+0.1)));G5=sqrt((P5)/((P11+0.1)));G6=sqrt((P6)/((P12+0.1)));
%   G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2)/((P8+P2)));G3=sqrt((P3)/((P9+P3)));
%   G4=sqrt((P4)/((P10+P4)));G5=sqrt((P5)/((P11+P5)));G6=sqrt((P6)/((P12+P6)));
% G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2)/((P8+P2)));G3=sqrt((P3)/((P9+P3)));
%   G4=sqrt((P4)/((P10+P4)));G5=sqrt((P5)/((P11+P5)));G6=sqrt((P6)/((P12+P6)));
  %%%%%%%���շ������ϳ�ͼ��
  RGB_06=imadd(0.9*RGB_0,G1*RGB_6);
  RGB_17=imadd(0.8*RGB_1,G2*RGB_7);
  RGB_28=imadd(0.7*RGB_2,G3*RGB_8);
  RGB_39=imadd(0.6*RGB_3,G4*RGB_9);
  RGB_410=imadd(0.5*RGB_4,G5*RGB_10);
  RGB_511=imadd(0.5*RGB_5,G6*RGB_11);
  figure('NumberTitle', 'off', 'Name', '���������Ч��');
  imshow(RGB_06);
    figure('NumberTitle', 'off', 'Name', '12=>=>HSV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
 subplot(231),imshow(RGB_06,[])  ,title('HSV6Ƶ�θ���');
 subplot(232),imshow(RGB_17,[])  ,title('HSV67Ƶ�θ���');
 subplot(233),imshow( RGB_28,[])  ,title('HSV678Ƶ�θ���');
 subplot(234),imshow( RGB_39,[])  ,title('HSV6789Ƶ�θ���');
subplot(235),imshow( RGB_410,[])  ,title('HSV678910Ƶ�θ���');
 subplot(236),imshow(RGB_511,[])  ,title('HSV67891011Ƶ�θ���');
 
%   figure('NumberTitle', 'off', 'Name', '���������Ч��');
%   imshow(RGB_06,[])  ,title('HSV6Ƶ�θ���');
  RGB_Q=imlincomb(1,RGB_06, 0.9,RGB_17, 0.8,RGB_28,0.7,RGB_39,0.6,RGB_410,0.5,RGB_511);
%   RGB_E=imlincomb(1,RGB_06, 1,RGB_17, 1,RGB_28,1,RGB_39,1,RGB_410,1,RGB_511);
  RGB_E=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,4,RGB_39,5,RGB_410,6,RGB_511);
%   RGB_E=imlincomb(1,RGB_06, 2,RGB_17, 5,RGB_28,4,RGB_39,2,RGB_410,1,RGB_511);
%     RGB_E=imlincomb(1,RGB_06, 5,RGB_17, 2,RGB_28,3,RGB_39,52,RGB_410,9,RGB_511);
%       RGB_Y=imlincomb(1,RGB_06, 9,RGB_17, 4,RGB_28,3,RGB_39,2,RGB_410,1,RGB_511);
%   %D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;     
%    subplot(231),imshow(RGB_Q,[])  ,title('RGB06');
 figure('NumberTitle', 'off', 'Name', '�𼶵ݼ��ĸ��ϱ���');
imshow(RGB_Q,[])  ,title('1:0.9:0.8:0.7:0.6:0.5');
figure('NumberTitle', 'off', 'Name', '��ͬ����');
RGB_W=imlincomb(1,RGB_06, 1,RGB_17, 1,RGB_28,1,RGB_39,1,RGB_410,1,RGB_511);
imshow(RGB_W,[])  ,title('1:1:1:1:1:1');
figure('NumberTitle', 'off', 'Name', '123432');
RGB_F=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,4,RGB_39,3,RGB_410,2,RGB_511);
imshow(RGB_F,[])  ,title('1:2:3:4:3:2');
figure('NumberTitle', 'off', 'Name', '����ٷֱȵ���');
RGB_T=imlincomb(0.1/G1,RGB_06, 0.1/G2,RGB_17, 0.1/G3,RGB_28,0.1/G4,RGB_39,0.1/G5,RGB_410,0.1/G6,RGB_511);
imshow(RGB_T,[])  ,title('10/(G1:G2:G3:G4:G5:G6)');
%  subplot(233),imshow( RGB_E,[])  ,title('1:1:1:1:1:1:1');
%  subplot(234),imshow( RGB_R,[])  ,title('1:2:3:4:5:6');
% subplot(235),imshow( RGB_T,[])  ,title('1:2:5:4:2:1');
%  subplot(236),imshow(RGB_Y,[])  ,title('152351');
%  figure('NumberTitle', 'off', 'Name', '��������������Ч������1:1:1:1:1:1:1�ϳ�');
%   imshow( RGB_E,[])  ,title('�������������水��1:1:1:1:1:1:1���');
%   figure('NumberTitle', 'off', 'Name', '���ķ�����棬����ƽ��');
%  %imshow( RGB_E,[])  ,title('ȫ������ƽ��');
% % imshow( RGB_E,[])  ,title('2345����ƽ��');
% %   subplot(233),imshow( RGB_E,[])  ,title('1345����ƽ��');
% %     subplot(234),imshow( RGB_E,[])  ,title('1245����ƽ��');
% %       subplot(235),imshow( RGB_E,[])  ,title('1235����ƽ��');
%    imshow( RGB_E,[])  ,title('����ƽ��ֻ��G1��');
     %imshow( RGB_E,[])  ,title('��ĸx10');
%  figure('NumberTitle', 'off', 'Name', '15=>139��ͬƵ��ǿ��������');
%  RGB_D=imlincomb(1,RGB_06, 0.5,RGB_17, 0.5,RGB_28,0.5,RGB_39,0.6,RGB_410,0.5,RGB_511);
%   RGB_I=imlincomb(1,RGB_06, 9,RGB_17, 3,RGB_28,3,RGB_39,3,RGB_410,3,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 3,RGB_17, 9,RGB_28,3,RGB_39,3,RGB_410,3,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,9,RGB_39,2,RGB_410,3,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 3,RGB_17,3,RGB_28,3,RGB_39,9,RGB_410,3,RGB_511);
%       RGB_M=imlincomb(1,RGB_06, 3,RGB_17, 3,RGB_28,3,RGB_39,2,RGB_410,9,RGB_511);

%   subplot(231),imshow(RGB_D,[])  ,title('1:0.5:0.5:0.5:0.5');
%  subplot(232),imshow(RGB_I,[])  ,title('1:933333');
%  subplot(233),imshow( RGB_C,[])  ,title('1:39333');
%  subplot(234),imshow( RGB_V,[])  ,title('1:33933');
% subplot(235),imshow( RGB_N,[])  ,title('13333393');
%  subplot(236),imshow(RGB_M,[])  ,title('1333339');
%  figure('NumberTitle', 'off', 'Name', '16=>124��ͬƵ��ǿ��������');
%  RGB_D=imlincomb(1,RGB_06, 4,RGB_17, 2,RGB_28,2,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_X=imlincomb(1,RGB_06, 2,RGB_17, 4,RGB_28,2,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,4,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,2,RGB_39,4,RGB_410,2,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 2,RGB_17,2,RGB_28,2,RGB_39,2,RGB_410,4,RGB_511);
%       RGB_M=imlincomb(0.85,RGB_06, 2,RGB_17, 2,RGB_28,2,RGB_39,2,RGB_410,4,RGB_511);

%   subplot(231),imshow(RGB_D,[])  ,title('142222');
%  subplot(232),imshow(RGB_X,[])  ,title('RGBI=>124444');
%  subplot(233),imshow( RGB_C,[])  ,title('=>122422');
%  subplot(234),imshow( RGB_V,[])  ,title('122242');
% subplot(235),imshow( RGB_N,[])  ,title('122224');
%  subplot(236),imshow(RGB_M,[])  ,title('1333339');
%  figure('NumberTitle', 'off', 'Name', '17=>124��ͬƵ��ǿ��������');
%  RGB_Z=imlincomb(1,RGB_06, 1.5,RGB_17, 3,RGB_28,1.5,RGB_39,1.5,RGB_410,1.5,RGB_511);
%   RGB_X=imlincomb(1,RGB_06, 3,RGB_17, 1.5,RGB_28,1.5,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,3,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 3,RGB_17, 4,RGB_28,4,RGB_39,5,RGB_410,2,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 2,RGB_17,2,RGB_28,2,RGB_39,2,RGB_410,3,RGB_511);
%       RGB_M=imlincomb(0.85,RGB_06, 1.5,RGB_17, 4,RGB_28,5,RGB_39,3,RGB_410,1,RGB_511);
% 
%   subplot(231),imshow(RGB_Z,[])  ,title('1;1.5;3');
%  subplot(232),imshow(RGB_X,[])  ,title('RGBI=>124444');
%  subplot(233),imshow( RGB_C,[])  ,title('=>122422');
%  subplot(234),imshow( RGB_V,[])  ,title('122242');
% subplot(235),imshow( RGB_N,[])  ,title('122224');
%  subplot(236),imshow(RGB_M,[])  ,title('1;1.5;4;5;3;1');
%  
 figure('NumberTitle', 'off', 'Name', '����ԭʼͼ��');
  subplot(121),imshow(RGB,[])  ,title('����ͼ��');
 subplot(122),imshow(RGB2,[])  ,title('���Ǩ��ͼ��');
%  figure('NumberTitle', 'off', 'Name', '����ԭʼͼ����Ǩ��ͼ��Ա�');
%   subplot(131),imshow(RGB,[])  ,title('����ͼ��');
%  subplot(132),imshow(RGB2,[])  ,title('���Ǩ��ͼ��');
%   subplot(133),imshow(RGB_V,[])  ,title('Ч��ͼ');

% clear all;
% close all;
% clc;
% 
% img=imread('timg.jpg');
% %    img=mat2gray(img);  %��������ӳ�䵽[0,1];
% 
% 
% tic
% R=img(:,:,1);                        %%ͼ���RGB
% G=img(:,:,2);
% B=img(:,:,3);
% [m n dim]=size(img);
% 
%              
% % Y=zeros(m,n);   %����%RGB2YUV
% % U=zeros(m,n);   %�ʶ�
% % V=zeros(m,n);   %Ũ��
% Y=0.299.*R+0.578.*G+0.114.*B;
% U=-0.1687.*R-0.3313.*G+0.5.*B+128;
% V=0.5.*R-0.4187.*G-0.0813.*B+128;
% yuv = cat(3, Y, U, V);  
% toc
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
%%%%%%%%%%%%%%%%%����ɫ�仯%%%%%%%%%%%%%%%%%%%%

% clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
%  clc
% % [R,G,B]=uigetfile('*.jpg','ѡ��ͼƬ');%%���Լ�ѡ��ͼƬ������
% clear%%ǧ����û��Ҫ��Ȼ����ЩĪ������Ĵ�
%  clc
%  [fn,pn,fi]=uigetfile('*.jpg','ѡ��ͼƬ');
%  tic
%  RGB=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', 'ԭͼ');%%figure������
% 
% 
% tic
% R=RGB(:,:,1);                        %%ͼ���RGB
% G=RGB(:,:,2);
% B=RGB(:,:,3);
% [m n dim]=size(RGB);
% 
%              
% Y=zeros(m,n);   %����%RGB2YUV
% U=zeros(m,n);   %�ʶ�
% V=zeros(m,n);   %Ũ��
% Y=0.299.*R+0.578.*G+0.114.*B;
% U=-0.1687.*R-0.3313.*G+0.5.*B+128;
% V=0.5.*R-0.4187.*G-0.0813.*B+128;
% yuv = cat(3, Y, U, V);  
% toc
% figure('NumberTitle', 'off', 'Name', 'YUVͼ��');
% imshow(yuv,[]);title('rgb2yuv');
% % tic
% % YUV=rgb2ycbcr(RGB);%ת��YUV
% % toc                                                              % % figure
%                                                                             % % imshow(YUV);
% 
% %%%����ͨ��%%%%%%%%%%
% Y=yuv(:,:,1);%ΪY��������
% U=yuv(:,:,2);%ΪU��������
% V=yuv(:,:,3);%ΪV��������
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
% yuv = cat(3, Y, U, V);  
% %%%%%%��ʾͼ��İ˹���%%%%%%%%
% figure('NumberTitle', 'off', 'Name', 'YUVͼ��Yͨ����Uͨ����Vͨ�� �ֱ���ʾ');%%figure������
% subplot(221),imshow(yuv),title('YUV');
% subplot(222),imshow(Y,[]),title('Y');
%  subplot(223),imshow(U),title('U');
%  subplot(224),imshow(V),title('V');
%  %%%%�ϳɲ��ԣ����������YUV��ʾͼһ����%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%���½��и���Ҷ�任%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   F=fft2(Y);          %����Ҷ�任
% %   F1=real(log(abs(F)+1));   %ȡģ���������� !!!!!�������ͼ����ɺں�ɫ
%   Fs=fftshift(F);%% �Ҳܲ���ȡģ��ȡģ����ĳ���Ӱ��Ҳ�Ծ���ֵ���ı���
%   %Fs=real(fftshift(F));      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
%                                         %%ע��Fs=F1,���fs=f,�ͳ�������
%                                         %%%����Fs=F����6���˲����˳���ͼ�ɼ�����Fs=F1���ɼ�                         
% %   S=log(abs(Fs)+1);    %ȡģ����������
% %   FFt= real(fftshift(F1));   %YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ
% %   Y=FFt;
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%��ԭ����YUV2RGB%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y=ifft2(ifftshift(Fs));
%  yuv= cat(3, Y,U,V); 
%  RGB_ = ycbcr2rgb(yuv);%ת��RGB
%  figure('NumberTitle', 'off', 'Name', 'YUVͼ��2RGB_');
%  imshow(RGB_);
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%�����Կ���������ԭRGBͼ��%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% figure('NumberTitle', 'off', 'Name', 'YUV��Yͨ���ĸ���ҶƵ��ͼ��YUVͼ��RGBͼ');
% subplot(131),imshow(Y,[])  ,title('Yͨ��Ƶ��ͼ');
% subplot(132),imshow(yuv,[])  ,title('YUV=>FFT��ͨ��ͼ');
% subplot(133),imshow(RGB_,[])  ,title('RGBͼ');
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%��������ͨ�˲���������˹�˲�%%%%%%%%%%
%  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
%  
%  n4=2;%step1%�˲����Ľ���2%%%%%%
%  %%step2%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
% D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
% %%%step3%%%%%%%6����ͨ�˲����Ľ�ֹƵ��%%%%%%%%%%%%%%%
%  [M,N]=size(F);%%%%%�˲�����С������ͼƬ%%%%%%%
% m=fix(M/2);
% n=fix(N/2);
% for i=1:M
%    for j=1:N
%         d=sqrt((i-m)^2+(j-n)^2);%%%%��㵽ͼ�����ľ���%%%%%%%  
%         
%         %%%%%%%%%%%%������˹��ͨ�˲���%%%%%%%%% 
%         h0=1/(1+0.414*(d/D0)^(2*n4)); %����D0=10;��ͨ�˲������ݺ���
%                s0(i,j)=h0*Fs(i,j);                   %%D0=10;�˲�����������ͼ��
%                T0(i, j) = h0;                           %%D0=10;�˲���������
%                
%         h1=1/(1+0.414*(d/D1)^(2*n4));%�����ͨ�˲������ݺ���
%               s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;�˲�����������ͼ��
%               T1(i, j) = h1;                          %%%%D1=20;�˲���������
%        
%         h2=1/(1+0.414*(d/D2)^(2*n4));%����D2=40;��ͨ�˲������ݺ���
%                s2(i,j)=h2*Fs(i,j);                  %%D2=40;�˲�����������ͼ��
%                T2(i, j) = h2;                         %%%%D2=40;�˲���������
%                
%         h3=1/(1+0.414*(d/D3)^(2*n4)); %����D3=60;��ͨ�˲������ݺ���
%                 s3(i,j)=h3*Fs(i,j);                  %D3=60;�˲�����������ͼ��
%                 T3(i, j) = h3;                         %%;D3=60;�˲���������
%                 
%         h4=1/(1+0.414*(d/D4)^(2*n4)); %����D4=80;��ͨ�˲������ݺ���
%                 s4(i,j)=h4*Fs(i,j);                  %%D4=80;�˲�����������ͼ��
%                 T4(i, j) = h4;                          %%D4=80;�˲���������
%                 
%         h5=1/(1+0.414*(d/D5)^(2*n4)); %����D5=255;��ͨ�˲������ݺ���
%                 s5(i,j)=h5*Fs(i,j);                  %%D5=255;�˲�����������ͼ��
%                 T5(i, j) = h5;                         %%D5=255;�˲���������
%        
%    end
% end
% 
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
% figure('NumberTitle', 'off', 'Name', '6����ͬƵ���˲������Ӽ�������ͼ��');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10��Ч��ͼ');%%D0=10;�˲�����������ͼ��
% subplot(3,4,2);imshow(T0);title('��ͨ�˲���D0=10');%%D0=10;�˲�������
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;�˲�����������ͼ��
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
% yuv1= cat(3, fr10, U, V);  
% yuv2 = cat(3, fr21, U, V);  
% yuv3 = cat(3, fr32, U, V);  
% yuv4 = cat(3,  fr43,U, V);  
% yuv5 = cat(3, fr54,U,V);  
% RGB_0= ycbcr2rgb(yuv0);%ת��RGB
%  RGB_1= ycbcr2rgb(yuv1);
%  RGB_2= ycbcr2rgb(yuv2);
%  RGB_3= ycbcr2rgb(yuv3);
%  RGB_4= ycbcr2rgb(yuv4);
%   RGB_5= ycbcr2rgb(yuv5);
% figure('NumberTitle', 'off', 'Name', 'YUV����6����ͬƵ���˲������Ӽ�������ͼ��');
% subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
% subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
% 
% RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
%  RGB_012=imadd(1*RGB_01,0.8*RGB_2);
%   RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
%    RGB_01234=imadd(1*RGB_0123,0.6*RGB_4);
%       RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
%        S=abs( RGB_012345)+1;    %ȡģ����������
%  figure('NumberTitle', 'off', 'Name', '6=>YUV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
%  
% 
%  subplot(231),imshow(RGB_0,[])  ,title('YUV-0Ƶ�θ���');
%  subplot(232),imshow(RGB_01,[])  ,title('YUV-01Ƶ�θ���');
%  subplot(233),imshow( RGB_012,[])  ,title('YUV-012Ƶ�θ���');
%  subplot(234),imshow( RGB_0123,[])  ,title('YUV-0123Ƶ�θ���');
% subplot(235),imshow( RGB_01234,[])  ,title('YUV-01234Ƶ�θ���');
%  subplot(236),imshow(RGB_012345,[])  ,title('YUV-012345Ƶ�θ���');
%  
%  yuv0 = cat(3, fr0,U,V);  
% yuv1= cat(3, fr10, U, V);  
% yuv2 = cat(3, fr21, U, V);  
% yuv3 = cat(3, fr32, U, V);  
% yuv4 = cat(3,  fr43,U, V);  
% yuv5 = cat(3, fr54,U,V);  
% RGB_0= ycbcr2rgb(yuv0);%ת��RGB
%  RGB_1= ycbcr2rgb(yuv1);
%  RGB_2= ycbcr2rgb(yuv2);
%  RGB_3= ycbcr2rgb(yuv3);
%  RGB_4= ycbcr2rgb(yuv4);
%   RGB_5= ycbcr2rgb(yuv5);
% figure('NumberTitle', 'off', 'Name', 'YUV����6����ͬƵ���˲������Ӽ�������ͼ��');
% subplot(3,4,1);imshow(yuv0,[]);title('D0=10��Ч��ͼ');
% subplot(3,4,2);imshow(T0,[]);title('��ͨ�˲���D0=10');
% subplot(3,4,3);imshow(yuv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0,[]),title('��ͨ�˲���D1=20 D1-D0');
% subplot(3,4,5);imshow(yuv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('��ͨ�˲���D2=40 D2-D1');
% subplot(3,4,7);imshow(yuv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('��ͨ�˲���D3=60 D3-D2');
% subplot(3,4,9);imshow(yuv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('��ͨ�˲���D4=60 D4-D3 ');
% subplot(3,4,11);imshow(yuv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4,[]),title('��ͨ�˲���D5=255 D5-D4');
% 
% RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J�Ƕ��������ͼ��
%  RGB_012=imadd(1*RGB_01,0.8*RGB_2);
%   RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
%    RGB_01234=imadd(1*RGB_0123,0.6*RGB_4);
%       RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
%        S=abs( RGB_012345)+1;    %ȡģ����������
%  figure('NumberTitle', 'off', 'Name', '6=>YUV��ԭRGB��ͬ6Ƶ���˲������Ӽ�������ͼ��');
%  
% 
%  
% imshow(RGB_012345,[])  ,title('6Ƶ�θ���');
% 