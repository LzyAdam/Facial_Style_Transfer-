RGB=imread('timg.jpg');%读入后便是一个RGB矩阵
YUV=rgb2ycbcr(RGB);%转成YUV

% %% %% %测试会不会得去% %% %% %% %% %% %% %% %
% % RGB_ = ycbcr2rgb(YUV);%转成RGB
% %  figure('NumberTitle', 'off', 'Name', '图1YUV2RGB');
% % subplot(121),imshow(RGB)  ,title('YUV2RGB回得去吗');
% %% %% %测试会不会得去% %% %% %% %% %% %% %% %

Y=YUV(:,:,1);
 U=YUV(:,:,2);
V= YUV(:,:,3);
% %显示Y-亮度通道的图像
figure('NumberTitle', 'off', 'Name', '图2Y-亮度图像是什么样子');
imshow(Y,[])  ,title('Y');
yuv=cat(3,Y,U,V);
figure('NumberTitle', 'off', 'Name', 'YUV图像，Y通道，Cb通道，Cr通道 分别显示');%%figure改名字
subplot(221),imshow(yuv),title('yuv');
subplot(222),imshow(Y,[]),title('Y');
 subplot(223),imshow(U),title('Cb');
 subplot(224),imshow(V),title('Cr');
%Y=im2double(Y);
 F=fft2(Y);          %傅里叶变换
 F1=log(abs(F)+1);   %取模并进行缩放%反变化一定要处理之前的F，切忌处理取模缩放的东西
  Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
  S=log(abs(Fs)+1);    %取模并进行缩放%反变化一定要处理之前的F，切忌处理取模缩放的东西
  Y=S;
   yuv=cat(3,Y,U,V);
  figure('NumberTitle', 'off', 'Name', '图3YCbCr 3通道合并');
 subplot(121),imshow(yuv)  ,title('合并三通道频谱');
subplot(122),imshow(Y,[])  ,title('Y通道频谱');%[]就能显示成黑色

 fr=real(ifft2(ifftshift(Fs)));  %频率域反变换到空间域，并取实部
 Y=fr;
 yuv=cat(3,fr,U,V);
   figure('NumberTitle', 'off', 'Name', 'YCbCr ifft');
imshow(yuv)  ,title('fr ifft');

 RGB_ = ycbcr2rgb(yuv);%转成RGB
 figure('NumberTitle', 'off', 'Name', 'YCbCr转RGB');
 imshow(RGB_)  ,title('YCbCr');
 
  fr=real(ifft2(ifftshift(Fs)));  %频率域反变换到空间域，并取实部
ret=im2uint8(mat2gray(fr));    %更改图像类型
figure('NumberTitle', 'off', 'Name', 'YCbCr转RGB和原图');
subplot(121),imshow(RGB_),title('YCbCr转RGB');
subplot(122),imshow(RGB),title('原图');

% %   
%    figure('NumberTitle', 'off', 'Name', 'Y1');
% subplot(131),imshow(Y1,[])  ,title('Y1');
%   yuv=cat(3,Y1,U,V);
%  % %测试会不会得去
% RGB_ = ycbcr2rgb(yuv);%转成RGB
%  figure('NumberTitle', 'off', 'Name', 'YUV2RGBnumber2');
% subplot(121),imshow(RGB)  ,title('YUV2RGB回得去吗');




% figure('NumberTitle', 'off', 'Name', '傅里叶变换频谱图');
% subplot(2,2,2);imshow(F1,[]);title('傅里叶变换频谱图');
% 
% figure('NumberTitle', 'off', 'Name', '傅里叶变换频谱图');
% subplot(2,2,3);imshow(S,[]);title('频移后的频谱图');
% fr=real(ifft2(ifftshift(Fs)));  %频率域反变换到空间域，并取实部
% ret=im2uint8(mat2gray(fr));    %更改图像类型
% subplot(2,2,4);imshow(ret),title('逆傅里叶变换');
% % % % % % %  figure('NumberTitle', 'off', 'Name', 'YUVfft');
% % % % % % % subplot(131),imshow(S,[])  ,title('Y通道频谱图');
% % % % % % %  subplot(132),imshow(yuv)  ,title('YUV');
% % % % % % %   subplot(133),imshow(RGB_)  ,title('RGB');
% 

%%%%%ifft傅里叶反变化
% %  [fn,pn,fi]=uigetfile('*.jpg','选择图片');
% % 
% %  img=imread([pn fn ]);
% % subplot(2,2,1);imshow(img);title('原图');
% % f=rgb2gray(img);    %对于RGB图像必须做的一步，也可以用im2double函数
% % F=fft2(f);          %傅里叶变换
% % F1=log(abs(F)+1);   %取模并进行缩放
% % subplot(2,2,2);imshow(F1,[]);title('傅里叶变换频谱图');
% % Fs=fftshift(F);      %将频谱图中零频率成分移动至频谱图中心
% % S=log(abs(Fs)+1);    %取模并进行缩放

% % fr=real(ifft2(ifftshift(Fs)));  %频率域反变换到空间域，并取实部
% % ret=im2uint8(mat2gray(fr));    %更改图像类型
% % subplot(2,2,4);imshow(ret),title('逆傅里叶变换');
% % subplot(2,2,3);imshow(fr,[]);title('频移后的频谱图');