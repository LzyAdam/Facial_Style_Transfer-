RGB=imread('timg.jpg');%��������һ��RGB����
YUV=rgb2ycbcr(RGB);%ת��YUV

% %% %% %���Ի᲻���ȥ% %% %% %% %% %% %% %% %
% % RGB_ = ycbcr2rgb(YUV);%ת��RGB
% %  figure('NumberTitle', 'off', 'Name', 'ͼ1YUV2RGB');
% % subplot(121),imshow(RGB)  ,title('YUV2RGB�ص�ȥ��');
% %% %% %���Ի᲻���ȥ% %% %% %% %% %% %% %% %

Y=YUV(:,:,1);
 U=YUV(:,:,2);
V= YUV(:,:,3);
% %��ʾY-����ͨ����ͼ��
figure('NumberTitle', 'off', 'Name', 'ͼ2Y-����ͼ����ʲô����');
imshow(Y,[])  ,title('Y');
yuv=cat(3,Y,U,V);
figure('NumberTitle', 'off', 'Name', 'YUVͼ��Yͨ����Cbͨ����Crͨ�� �ֱ���ʾ');%%figure������
subplot(221),imshow(yuv),title('yuv');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
%Y=im2double(Y);
 F=fft2(Y);          %����Ҷ�任
 F1=log(abs(F)+1);   %ȡģ����������%���仯һ��Ҫ����֮ǰ��F���мɴ���ȡģ���ŵĶ���
  Fs=fftshift(F);      %��Ƶ��ͼ����Ƶ�ʳɷ��ƶ���Ƶ��ͼ����
  S=log(abs(Fs)+1);    %ȡģ����������%���仯һ��Ҫ����֮ǰ��F���мɴ���ȡģ���ŵĶ���
  Y=S;
   yuv=cat(3,Y,U,V);
  figure('NumberTitle', 'off', 'Name', 'ͼ3YCbCr 3ͨ���ϲ�');
 subplot(121),imshow(yuv)  ,title('�ϲ���ͨ��Ƶ��');
subplot(122),imshow(Y,[])  ,title('Yͨ��Ƶ��');%[]������ʾ�ɺ�ɫ

 fr=real(ifft2(ifftshift(Fs)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
 Y=fr;
 yuv=cat(3,fr,U,V);
   figure('NumberTitle', 'off', 'Name', 'YCbCr ifft');
imshow(yuv)  ,title('fr ifft');

 RGB_ = ycbcr2rgb(yuv);%ת��RGB
 figure('NumberTitle', 'off', 'Name', 'YCbCrתRGB');
 imshow(RGB_)  ,title('YCbCr');
 
  fr=real(ifft2(ifftshift(Fs)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
ret=im2uint8(mat2gray(fr));    %����ͼ������
figure('NumberTitle', 'off', 'Name', 'YCbCrתRGB��ԭͼ');
subplot(121),imshow(RGB_),title('YCbCrתRGB');
subplot(122),imshow(RGB),title('ԭͼ');

% %   
%    figure('NumberTitle', 'off', 'Name', 'Y1');
% subplot(131),imshow(Y1,[])  ,title('Y1');
%   yuv=cat(3,Y1,U,V);
%  % %���Ի᲻���ȥ
% RGB_ = ycbcr2rgb(yuv);%ת��RGB
%  figure('NumberTitle', 'off', 'Name', 'YUV2RGBnumber2');
% subplot(121),imshow(RGB)  ,title('YUV2RGB�ص�ȥ��');




% figure('NumberTitle', 'off', 'Name', '����Ҷ�任Ƶ��ͼ');
% subplot(2,2,2);imshow(F1,[]);title('����Ҷ�任Ƶ��ͼ');
% 
% figure('NumberTitle', 'off', 'Name', '����Ҷ�任Ƶ��ͼ');
% subplot(2,2,3);imshow(S,[]);title('Ƶ�ƺ��Ƶ��ͼ');
% fr=real(ifft2(ifftshift(Fs)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
% ret=im2uint8(mat2gray(fr));    %����ͼ������
% subplot(2,2,4);imshow(ret),title('�渵��Ҷ�任');
% % % % % % %  figure('NumberTitle', 'off', 'Name', 'YUVfft');
% % % % % % % subplot(131),imshow(S,[])  ,title('Yͨ��Ƶ��ͼ');
% % % % % % %  subplot(132),imshow(yuv)  ,title('YUV');
% % % % % % %   subplot(133),imshow(RGB_)  ,title('RGB');
% 

%%%%%ifft����Ҷ���仯
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

% % fr=real(ifft2(ifftshift(Fs)));  %Ƶ���򷴱任���ռ��򣬲�ȡʵ��
% % ret=im2uint8(mat2gray(fr));    %����ͼ������
% % subplot(2,2,4);imshow(ret),title('�渵��Ҷ�任');
% % subplot(2,2,3);imshow(fr,[]);title('Ƶ�ƺ��Ƶ��ͼ');