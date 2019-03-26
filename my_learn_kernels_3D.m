%此代码目的是将边界处理加入到3D代码中：思路是根据2D-3D代码修改3D代码
%按照FFCSC的方式，将边界处理加入。
%但是由于不进行consensus，这里我只
imgs_path = '..\..\Hyperspectral DATA\LEGO\';
load([imgs_path 'training_data.mat'], 'b');
size(b)%256 256 24



