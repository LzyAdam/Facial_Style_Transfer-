
% % [fn,pn,fi]=uigetfile({'*.bmp' ;'*.jpg'},'ѡ��ͼƬ');
% % 
% % I=imshow([pn fn ]);
%subplot(221);
%b=fft2(I);
%c=imshow(b);
% I=imread([pn fn fi]);
% I=rgb2yuv(I);
% I=rgb2gray(I);
% %=im2double(I);
% F=fft2(I);
% F=fftshift(F);
% F=abs(F);
% T=log(F+1);
% figure;
% 
% % % A=imshow(T,[]);
% % %  B=sumsqr(F);
% % % disp(B);
% %  A=[0,1,2,3;4,5,6,7]
% %  
% %  sumsqr(A)
% % 
% %  

% % [m n dim]=uigetfile({'*.bmp' ;'*.jpg'},'ѡ��ͼƬ');
% % % % 
% % I=imshow([m n dim ]);
% % img=im2double(I);
% % % % img=imread('lena_color.jpg');
% % img=mat2gray(img);  %��������ӳ�䵽[0,1];
% % [m n dim]=size(img);
% % imshow(img);
% % %%ͼ���RGB
% % R=img(:,:,1);
% % G=img(:,:,2);
% % B=img(:,:,3);
% % 
% % %%RGB2YUV
% % Y=zeros(m,n);   %����
% % U=zeros(m,n);   %�ʶ�
% % V=zeros(m,n);   %Ũ��
% % matrix=[0.299 0.587 0.114;
% %         -0.14713 -0.28886 0.436;
% %         0.615 -0.51498 -0.10001];
% % for i=1:m
% %    for j=1:n 
% %         tmp=matrix*[R(i,j) G(i,j) B(i,j)]';
% %         Y(i,j)=tmp(1);
% %         U(i,j)=tmp(2);
% %         V(i,j)=tmp(3);
% %    end
% % end

% % %%YUV2RGB
% % matrix=inv(matrix);
% % for i=1:m
% %    for j=1:n 
% %         tmp=matrix*[Y(i,j) U(i,j) V(i,j)]';
% %         R(i,j)=tmp(1);
% %         G(i,j)=tmp(2);
% %         B(i,j)=tmp(3);
% %    end
% % end
% % 
% % %%��������任��û��Ļ�����ôͼ���ǲ����
% % img(:,:,1)=R;
% % img(:,:,2)=G;
% % img(:,:,3)=B;
% % figure;
% % imshow(img)



% % 
% % 
% % clear all;
% % close all;
% % clc;
% % 
img=imread('timg.jpg');
% img=mat2gray(img);  %��������ӳ�䵽[0,1];


%%ͼ���RGB
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
[m n dim]=size(img);

%%RGB2YUV
Y=zeros(m,n);   %����
U=zeros(m,n);   %�ʶ�
V=zeros(m,n);   %Ũ��
Y=0.299*R+ 0.587*G+ 0.114*B

U=-0.147*R-0.289*G+0.436*B
V=0.615*R-0.515*G-0.100*B

% % 
% % %I=rgb2gray(Y);  %���в�ȻͼƬ�ᱻѹ����һ���С
% % % %����RGBͼ���������һ����Ҳ������im2double����
% % I=im2double(Y);
% %  F=fft2(I);          %����Ҷ�任
% %  F1=log(abs(F)+1);   %ȡģ����������
% % 
% %  Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
% %  S=log(abs(Fs)+1);    %ȡģ����������
% % Y=S;
% %  imshow(Y,[])
 yuv = cat(3, Y, U, V);  
% %  rgb = cat(3, R, G, B);  
% %  
figure('NumberTitle', 'off', 'Name', 'YUVͼ��Yͨ����Uͨ����Vͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('yuv');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('U');
 subplot(224),imshow(V),title('V');
 
 RGB_ = ycbcr2rgb(yuv);%ת��RGB
 figure
subplot(111),imshow(RGB_ ,[]),title('RGB');
 
figure('NumberTitle', 'off', 'Name', 'RGBͼ���YUVͼ��Ա�');%%figure������
 subplot(121),imshow(RGB_),title('RGB');
  subplot(122),imshow(yuv),title('YUV');
% % 
% % figure('NumberTitle', 'off', 'Name', 'RGBͼ��Rͨ����Gͨ����Bͨ���ֱ���ʾ');%%figure������
% % 
% %   subplot(221),imshow(rgb),title('rgb');
% % subplot(222),imshow(R),title('R');
% %  subplot(223),imshow(G),title('G');
% %  subplot(224),imshow(B),title('B');
 
% % % YUV2RGB
% % % matrix=inv(matrix);
% % % for i=1:m
% % %    for j=1:n 
% % %         tmp=matrix*[Y(i,j) U(i,j) V(i,j)]';
% % %         R(i,j)=tmp(1);
% % %         G(i,j)=tmp(2);
% % %         B(i,j)=tmp(3);
% % %    end
% % % end
% % % 
% % % %%%%��������任��û��Ļ�����ôͼ���ǲ����
% % % img(:,:,1)=R;
% % % img(:,:,2)=G;
% % % img(:,:,3)=B;
% % % figure;
% % % imshow(img)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %  I=imread([pn fn ]);

% % RGB=imread('timg.jpg');%��������һ��RGB����
% % YUV=rgb2ycbcr(RGB);%ת��YUV
% %  RGB_ = ycbcr2rgb(YUV);%ת��RGB
% %  figure('NumberTitle', 'off', 'Name', 'RGB');
% % subplot(131),imshow(RGB)  ,title('Yͨ��Ƶ��ͼ');
% % % % Y=imread([r g ]);
% % Y=YUV(:,:,1);
% % % % YUV(:,:,3)
% %  U=YUV(:,:,2);
% % V= YUV(:,:,3);
% % figure('NumberTitle', 'off', 'Name', 'Y');
% % subplot(131),imshow(Y,[])  ,title('Y');
% % subplot;
% %  A=imshow(P);
% % % % ����RGBͼ���������һ����Ҳ������im2double����
%%Y=im2double(Y);%%%�������ͼ���д���䰵

% %  F=fft2(Y);          %����Ҷ�任
% %  % % F1=log(abs(F)+1);   %ȡģ����������
% %  Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
% % % %   S=log(abs(Fs)+1);    %ȡģ����������
% %   T=ifftshift(Fs);
% %   Y1=ifft2(T);
% %    figure('NumberTitle', 'off', 'Name', 'Y1');
% % subplot(131),imshow(Y1,[])  ,title('Y1');
% %   yuv=cat(3,Y1,U,V);
% %   RGB_ = ycbcr2rgb(yuv);%ת��RGB
% %  figure('NumberTitle', 'off', 'Name', 'YUVfft');
% % subplot(131),imshow(S,[])  ,title('Yͨ��Ƶ��ͼ');
% %  subplot(132),imshow(yuv)  ,title('YUV');
% %   subplot(133),imshow(RGB_)  ,title('RGB');

% subplot(2,2,2);imshow(T1);title('��ͨ�˲���');

% % RGB_ = ycbcr2rgb(YUV);%ת��RGB
% % subplot
% %  imshow(RGB_);
 
 %RGB(:,:,1)ΪR��������
%RGB(:,:,2)ΪG��������
%RGB(:,:,3)ΪB��������
%YUV(:,:,1)ΪY��������
%YUV(:,:,2)ΪU��������
%YUV(:,:,3)ΪV��������

%��ʽ Y = 0.2990*R + 0.5780*G + 0.1140*B + 0
%��ʽ U = 0.5000*R - 0.4187*G - 0.0813*B + 128
%��ʽ V = -0.1687*R - 0.3313*G + 0.5000*B + 128