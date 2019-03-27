%�˴���Ŀ���ǽ��߽紦����뵽3D�����У�˼·�Ǹ���2D-3D�����޸�3D����
%����FFCSC�ķ�ʽ�����߽紦����롣
%�������ڲ�����consensus��������ֻ
clc,clear
imgs_path = '..\..\Hyperspectral DATA\LEGO\';
load([imgs_path 'training_data.mat'], 'b');
size(b);%��СΪ256 256 24
size(b,4);%1
%Ϊʲô2D-3D�Ĵ����ǽ�����ά�ȵĴ�С����Ϊ��ԭͼ�����λ�����Ϊ��2D-3D������
%���о����ʱ�����ڶ�ά�Ͻ��о����
kernel_size=[11,11,3,20];%kernel_size�Ĵ�С��ԭ����һ����
lambda_residual=1.0;
lambda=1.0;
verbose='all';
%ǰ��Ķ���ͼ���Ԥ�����2D-3Dһ��
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(b, k, 'same', 'conv', 'symmetric');
size(smooth_init);%256 256 24 1
% figure();
% for i=1:size(smooth_init,3)
%     subplot(121)
%     imshow(b(:,:,i,1)),title(sprintf('number of image:%d',i));
%     subplot(122)
%     imshow(smooth_init(:,:,i,1)),title('�����˸�˹����');
%     pause(0.5);
% end
%����˵����ʹ�������˸�˹������ԭ����ȥ����ѧϰfiltersȻ��ȥ�ؽ���
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d x %d] kernels.\n\n', kernel_size(4), kernel_size(1), kernel_size(2), kernel_size(3))

verbose_admm = 'brief';
max_it = 100;%40;%max_it = 3000;%60; %  
tol = 1e-4;% 1e-3; %3D�Ǳ����õ���1e-2�����Ի�����ĸ��졣
init = [];

tic();
[d, z, Dz, obj]  = my3D_admm_learn(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, init, smooth_init);
tt = toc;

% Debug
fprintf('Done dictionary learning! --> Time %2.2f sec.\n\n', tt)

% Save dictionary
save('./my3D-Hyperspectral.mat', 'd', 'z', 'Dz', '-v7.3');
