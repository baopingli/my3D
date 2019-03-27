function [ d_res, z_res, Dz, obj_val ] = my3D_admm_learn(b, kernel_size,...
    lambda_residual, lambda_prior, ...
    max_it, tol, ...
    verbose, init, smooth_init)
%3D的仿照2D-3D代码写的。后面才是难的
%首先是声明一些size
psf_s1=kernel_size(1);%11,但是这现在是3D了所以说对于第三维度也需要psf
psf_s2=kernel_size(2);%11
psf_s3=kernel_size(3);%3
k=kernel_size(4);%20

n=size(b,4);%number of samples. 

ni=sqrt(n);%1
N=n/ni;%1

psf_radius1=floor(psf_s1/2);
psf_radius2=floor(psf_s2/2);
psf_radius3=floor(psf_s3/2);
psf_radius=[psf_radius1,psf_radius2,psf_radius3];

size_x=[size(b,1)+2*psf_radius1,size(b,2)+2*psf_radius2,size(b,3)+2*psf_radius3,n];
%266 266 26 1
size_z=[size_x(1:end-1),k,n];
%266 266 26 20 1
size_z_crop=[size_x(1:end-1),k,ni];%估计用不到size_z_crop，只用到了size_z
%266 266 26 20 1
size_d_full=[size_x(1:end-1),k];
%266 266 26 20
size_k_full=[size_x(1),size_x(2),size_x(3),k]
%好像和size_d_full大小相等， 266 266 26 20
size_zhat=size_z;
%266 266 26 20 1
smoothinit=padarray(smooth_init,[psf_radius1,psf_radius2,psf_radius3,0], 'symmetric', 'both');
%266 266 26

%objective
objective = @(z, dh) objectiveFunction( z, dh, b, lambda_residual, lambda_prior, psf_radius, size_z, size_x, smoothinit );
%prox for masked data
[M, Mtb] = precompute_MProx(b, psf_radius, smoothinit);
ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) );
%prox for sparsity
ProxSparse = @(u, theta) max( 0, 1 - theta./ abs(u) ) .* u;
%Prox for kernel constraints
ProxKernelConstraint = @(u) KernelConstraintProj( u, size_k_full, psf_radius);

lambda=[lambda_residual,lambda_prior];
gamma_heuristic = 60 * lambda_prior * 1/max(b(:));
gammas_D = [gamma_heuristic, gamma_heuristic];
gammas_Z = [gamma_heuristic, gamma_heuristic];

varsize_D={size_x, size_k_full};%zd size and filter size(zd d)
xi_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % intermediate variable (u_D+d_D) for the quadratic problem of filters d.
xi_D_hat = { zeros(varsize_D{1}), zeros(varsize_D{2}) };

u_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % slack variables in terms of primary variable (filters d): {u_D{1}=Zd, u_D{2}=d}.
d_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % dual variables.
v_D = { zeros(varsize_D{1}), zeros(varsize_D{2}) }; % intermediate variables for storing the values of { Zd, d}

%Initial iterates 初始化d
if ~isempty(init)
    d_hat = init.d;
    d = [];
else % pad filters to the full convolution dimensions and circularly shift to account for the boundary effect.
    %         d = padarray( randn(kernel_size([1 2 5])), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), 0], 0, 'post');%
    d = padarray( randn(kernel_size([1 2 3])), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2), size_x(3)-kernel_size(3)], 0, 'post');% inititialize filters to random noise
    d = circshift(d, -[psf_radius(1), psf_radius(2), psf_radius(3)] );
    %         d = permute(repmat(d, [1 1 1 kernel_size(3) kernel_size(4)]), [1 2 4 5 3]);
    d = repmat(d, [1 1 1 k]); % (x,y,spectral, filter)
    d_hat = fft2(d);
    %所以说d_hat的大小为 266 266 26 20 
end

%初始化z的相关变量
varsize_Z={size_x,size_z};%Dz  z
xi_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
xi_Z_hat = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };

u_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };
d_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) }; % dual variables.
v_Z = { zeros(varsize_Z{1}), zeros(varsize_Z{2}) };% intermediate variables for storing the values of { Dz, z}
%从现在的观察来看，我发现不同的地方是系数z多了第三维度
z = randn(size_z);
obj_val = objective(z, d_hat);
if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
    fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
end
max_it_d = 10;
max_it_z = 10;
obj_val_filter = obj_val;
obj_val_z = obj_val;
for i=1:max_it
    rho=gammas_D(2)/gammas_D(1);
    obj_val_min = min(obj_val_filter, obj_val_z);
    d_old = d;
    d_hat_old = d_hat;
    z_hat = fft2(z);
    [zhatT_blocks, invzhatTzhat_blocks] = myprecompute_Z_hat_d(z_hat, gammas_D);
    for i_d=1:max_it_d
        %v_D是中间变量vD1是dz
        %首先需要知道z_hat和d_hat的大小，
        %根据我的观察这里好像是不需要进行reshape了
        %v_D{1} = real(ifft2( reshape(sum( bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4), size_x) ));% d_hat : (x,y,spectral,filters),
        v_D{1} = real(ifft2( reshape(sum( bsxfun(@times, d_hat, z_hat), 4), size_x) ));
        % z_hat : (x,y,filters,examples)  ,  permute(z_hat, [1 2 5 3 4]) : (x,y,spectral(1),filters,,examples).
        v_D{2}=d;
        u_D{1} = ProxDataMasked( v_D{1} - d_D{1}, lambda(1)/gammas_D(1) );
        u_D{2} = ProxKernelConstraint( v_D{2} - d_D{2});%这个只有一个就是在D更新的时候
        for c=1:2
            d_D{c} = d_D{c} - (v_D{c} - u_D{c});
            xi_D{c} = u_D{c} + d_D{c};
            xi_D_hat{c} = fft2(xi_D{c});
        end
        d_hat=mysolve_conv_term_D(zhatT_blocks, invzhatTzhat_blocks, xi_D_hat, gammas_D, size_k_full, n);
        d = real(ifft2( d_hat ));
        obj_val = objective(z, d_hat);
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('--> Obj %5.5g \n', obj_val )
        end        
        
    end
    obj_val_filter = obj_val;
    d_diff = d - d_old;
    d_comp = d;
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        obj_val = objective(z, d_hat);
        fprintf('Iter D %d, Obj %5.5g, Diff %5.5g\n', i, obj_val, norm(d_diff(:),2)/ norm(d_comp(:),2))
    end
    %update z
    %上来就计算矩阵逆
    [dhatT_blocks, invdhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas_Z);
    z_hat=fft2(z);
    z_old=z;
    z_hat_old=z_hat;
    for i_z=1:max_it_z
        %v_Z{1} = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4))));% Dz
        v_Z{1} = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, z_hat), 4))));% Dz
        v_Z{2} = z;
        u_Z{1} = ProxDataMasked( v_Z{1} - d_Z{1}, lambda(1)/gammas_Z(1) );
        u_Z{2} = ProxSparse( v_Z{2} - d_Z{2}, lambda(2)/gammas_Z(2) );
        for c=1:2
            d_Z{c} = d_Z{c} - (v_Z{c} - u_Z{c});
            
            %Compute new xi and transform to fft
            xi_Z{c} = u_Z{c} + d_Z{c};
            xi_Z_hat{c} = fft2( xi_Z{c} );
        end
        z_hat=mysolve_conv_term_Z(dhatT_blocks, invdhatTdhat_blocks, xi_Z_hat, gammas_Z, size_z, kernel_size(3));
        z=real(ifft2(z_hat));
        obj_val = objective(z, d_hat);%这里有错，非单一维度要相互匹配
        if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
            fprintf('--> Obj %5.5g \n', obj_val )
        end
    end
    obj_val_z = obj_val;
    z_diff = z - z_old;
    z_comp = z;
    if strcmp(verbose, 'brief') || strcmp( verbose, 'all')
        fprintf('Iter Z %d, Obj %5.5g, Diff %5.5g, Sparsity %5.5g\n', i, obj_val, norm(z_diff(:),2)/ norm(z_comp(:),2), nnz(z(:))/numel(z(:)))
    end
    
    %Termination
    if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol && norm(d_diff(:),2)/ norm(d_comp(:),2) < tol
        break;
    end
end
%Final estimate
z_res = z;

d_res = circshift( d, [psf_radius(1), psf_radius(2), psf_radius(3), 0] );
d_res = d_res(1:psf_radius(1)*2+1, 1:psf_radius(2)*2+1, 1:psf_radius(3)*2+1, :);

z_hat = reshape(fft2(z), size_zhat);
%Dz = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, permute(z_hat, [1 2 5 3 4])), 4)))) + smoothinit;% Dz
Dz = real(ifft2(squeeze(sum(bsxfun(@times, d_hat, z_hat), 4)))) + smoothinit;

return;


function z_hat = mysolve_conv_term_Z(dhatT_blocks, invdhatTdhat_blocks, xi_hat, gammas, size_z, sw )
sy=size_z(1);
sx=size_z(2);
sz=size_z(3);
k=size_z(4);
n=size_z(5);
z_hat=zeros(k,n,sx*sy*sz);
x1_blocks=permute(reshape(xi_hat{1},sy*sx*sz,n),[2,1]);
x2_blocks=permute(reshape(xi_hat{2},sy*sx*sz,k,n),[2,3,1]);
rho=gammas(2)/gammas(1);
for i=1:sx*sy*sz
    z_hat(:,:,i)=invdhatTdhat_blocks(:,:,i) * ( dhatT_blocks(:,i) * x1_blocks(:,i) + rho * x2_blocks(:,:,i) );
end
%这里少了一句话
z_hat = reshape(permute(z_hat, [3,1,2]), size_z);%这样应该是对了

return;



function [dhatT_blocks, invdhatTdhat_blocks] = myprecompute_H_hat_Z(d_hat, gammas)

sy=size(d_hat,1);sx=size(d_hat,2);sw=size(d_hat,3);k=size(d_hat,4);
invdhatTdhat_blocks = zeros(k,k,sx * sy * sw);
rho=gammas(2)/gammas(1);
dhatT_blocks = conj( permute( reshape(d_hat, sx * sy * sw, k), [2,1]) ); 
%permute( reshape(d_hat, sx * sy, sw, k), [2,1,3]) rearranges d_hat into dimension sw*k*(sx*sy).
%应该首先将其转换为元胞然后使用元胞函数进行处理。所以说简单的方法是学会cellfun怎么用，然后在这上面改。
%将函数应用到所有的元胞上。
for i=1:sx*sy*sw
    invdhatTdhat_blocks(:,:,i) = pinv(rho * eye(k) + dhatT_blocks(:,i) * dhatT_blocks(:,i)');% inv[dTd_(i)_rho*I]
end
    
return;

function d_hat = mysolve_conv_term_D(zhatT_blocks, invzhatTzhat_blocks, xi_hat, gammas, size_d, n)

sy=size_d(1);sx=size_d(2);
sw=size_d(3);%第三维度26
k=size_d(4);%num of filters
%按着自己的想法写就是了
d_hat=zeros(k,sw*sx*sy);
%这个和3D代码中类似
x1_blocks=permute(reshape(xi_hat{1},sy*sx*sw,n),[2,1]);
x2_blocks=permute(reshape(xi_hat{2},sy*sx*sw,k),[2,1]);
rho=gammas(2)/gammas(1);
for i=1:sx*sy*sw
    d_hat(:,i)=invzhatTzhat_blocks(:,:,i)*(zhatT_blocks(:,:,i)*x1_blocks(:,i)+rho*x2_blocks(:,i));
end
d_hat=reshape(permute(d_hat,[2,1]),size_d);
return;




%计算dTd+rho*I,没有使用矩阵逆的引理，直接求的逆。(为什么耗的时间那么长)，可能真的需要特殊处理
function [zhatT_blocks, invzhatTzhat_blocks] = myprecompute_Z_hat_d(z_hat, gammas)
%就是把三个维度都算进去 如果这里不确定后面问问老师
sy=size(z_hat,1);sx=size(z_hat,2);sz=size(z_hat,3);k=size(z_hat,4);n=size(z_hat,5);
invzhatTzhat_blocks = zeros(k,k,sx * sy * sz);
rho=gammas(2)/gammas(1);
%上来就将矩阵进行了reshape处理
%  permute( reshape(z_hat, sx * sy * sz, k, n), [2,3,1]) rearranges z_hat into dimension k*n*(sx*sy).
zhatT_blocks=conj(permute(reshape(z_hat,sx*sy*sz,k,n),[2,3,1]));
for i=1:sx*sy*sz %这个数太大了导致运行的时间很长。（所以说3D代码好像是使用了bsxfun快速计算）
    invzhatTzhat_blocks(:,:,i) = pinv(rho * eye(k) + zhatT_blocks(:,:,i) * zhatT_blocks(:,:,i)');% inv[zTz_(i)+rho*I]
end
return;


% function [u_proj]=KernelConstraintProj( u, size_k_full, psf_radius)
% %Get support
% u_proj = circshift( u, [psf_radius(1), psf_radius(2), psf_radius(3), 0] );
% u_proj = u_proj(1:psf_radius(1)*2+1, 1:psf_radius(2)*2+1, 1:psf_radius(3)*2+1,:);
% 
% %Normalize
% u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), size(u_proj,3), 1] );
% %扩充到三个维度的大小，将sum的值，应该是这样对应到三个维度里面去实现。
% u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));
% %Now shift back and pad again 变回原状
% u_proj = padarray( u_proj, (size_k_full - size(u_proj)), 0, 'post');
% u_proj = circshift(u_proj, -[psf_radius(1), psf_radius(2), psf_radius(3), 0] );
% return;
function [u_proj] = KernelConstraintProj( u, size_d, psf_radius)

    %Params
    k = size_d(end);
    ndim = length( size_d ) - 1;

    %Get support
    u_proj = circshift( u, [psf_radius(1), psf_radius(2), psf_radius(3), 0] ); 
    u_proj = u_proj(1:psf_radius(1)*2+1,1:psf_radius(2)*2+1,1:psf_radius(3)*2+1,:);
    
     %Normalize
    for ik = 1:k
        u_curr = eval(['u_proj(' repmat(':,',1,ndim), sprintf('%d',ik), ')']);
        u_norm = sum(reshape(u_curr.^2 ,[],1));
        if u_norm >= 1
            u_curr = u_curr ./ sqrt(u_norm);
        end
        eval(['u_proj(' repmat(':,',1,ndim), sprintf('%d',ik), ') = u_curr;']);
    end
    
    %Now shift back and pad again
    %u_proj = padarray( u_proj, [size_d(1:end - 1) - (2*psf_radius+1), 0], 0, 'post');
    u_proj = padarray( u_proj, [size_d(1)-(2*psf_radius(1)+1),size_d(2)-(2*psf_radius(2)+1),size_d(3)-(2*psf_radius(3)+1), 0], 0, 'post');
    %u_proj = circshift(u_proj, -[repmat(psf_radius, 1, ndim), 0]);
    u_proj = circshift(u_proj, -[psf_radius(1), psf_radius(2), psf_radius(3), 0] );
    
return;


function [M, Mtb] = precompute_MProx(b, psf_radius,smoothinit)
M = padarray(ones(size(b)), [psf_radius(1), psf_radius(2), psf_radius(3), 0]);
Mtb = padarray(b, [psf_radius(1), psf_radius(2), psf_radius(3), 0]).*M - smoothinit.*M;

return;





function f_val = objectiveFunction( z, d_hat, b, lambda_residual, lambda, psf_radius, size_z, size_x, smoothinit)
    %z_hat=permute(fft2(z),[1,2,3,4,5]);%(x,y,z,filter,example) 
    % 266 266 26 20 1
    z_hat=fft2(z);
    %这个还需要看一下d_hat是什么格式 d_hat的 大小为 266 266 26 20，这样算的话应该没有问题不会出错
    Dz=real(ifft2(squeeze(sum(bsxfun(@times,d_hat,z_hat),4)) ))+smoothinit;
    %这个计算的是沿第四个维度的总和，这样应该可以，因为有filters那一维度
    f_z = lambda_residual * 1/2 * norm( reshape( Dz(1 + psf_radius(1):end - psf_radius(1), ...
    1 + psf_radius(2):end - psf_radius(2),1 + psf_radius(3):end - psf_radius(3),:) - b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );

%Function val
f_val = f_z + g_z;
return;