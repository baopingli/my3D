function [ d_res, z_res, Dz, obj_val ] = my3D_admm_learn(b, kernel_size,...
    lambda_residual, lambda_prior, ...
    max_it, tol, ...
    verbose, init, smooth_init)
%3D�ķ���2D-3D����д�ġ���������ѵ�
%����������һЩsize
psf_s1=kernel_size(1);%11,������������3D������˵���ڵ���ά��Ҳ��Ҫpsf
psf_s2=kernel_size(2);%11
psf_s3=kernel_size(3);%3
k=kernel_size(4);%20
n=size(b,4);%number of samples. ��������ò���
psf_radius1=floor(psf_s1/2);
psf_radius2=floor(psf_s2/2);
psf_radius3=floor(psf_s3/2);
size_x=[size(b,1)+2*psf_radius1,size(b,2)+2*psf_radius2,size(b,3)+2*psf_radius3,n];

end
