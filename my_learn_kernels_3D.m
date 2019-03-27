%此代码目的是将边界处理加入到3D代码中：思路是根据2D-3D代码修改3D代码
%按照FFCSC的方式，将边界处理加入。
%但是由于不进行consensus，这里我只
clc,clear
imgs_path = '..\..\Hyperspectral DATA\LEGO\';
load([imgs_path 'training_data.mat'], 'b');
size(b);%大小为256 256 24
size(b,4);%1
%为什么2D-3D的代码是将第三维度的大小设置为和原图像第三位相等因为是2D-3D，真正
%进行卷积的时候是在二维上进行卷积。
kernel_size=[11,11,3,20];%kernel_size的大小和原来不一样了
lambda_residual=1.0;
lambda=1.0;
verbose='all';
%前面的对于图像的预处理和2D-3D一致
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(b, k, 'same', 'conv', 'symmetric');
size(smooth_init);%256 256 24 1
% figure();
% for i=1:size(smooth_init,3)
%     subplot(121)
%     imshow(b(:,:,i,1)),title(sprintf('number of image:%d',i));
%     subplot(122)
%     imshow(smooth_init(:,:,i,1)),title('增加了高斯噪声');
%     pause(0.5);
% end
%所以说可以使用增加了高斯噪声的原数据去进行学习filters然后去重建。
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d x %d] kernels.\n\n', kernel_size(4), kernel_size(1), kernel_size(2), kernel_size(3))

verbose_admm = 'brief';
max_it = 100;%40;%max_it = 3000;%60; %  
tol = 1e-4;% 1e-3; %3D那边设置的是1e-2，所以会结束的更快。
init = [];

tic();
[d, z, Dz, obj]  = my3D_admm_learn(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, init, smooth_init);
tt = toc;

% Debug
fprintf('Done dictionary learning! --> Time %2.2f sec.\n\n', tt)

% Save dictionary
save('./my3D-Hyperspectral.mat', 'd', 'z', 'Dz', '-v7.3');
