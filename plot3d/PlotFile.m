clear; close all; clc;

filename = 'vector_data.csv';
info = readmatrix('dim.csv');
T = reshape(readmatrix(filename), info(1:3));
T = permute(T, [2 3 1]);
T = T(2:end-1,2:end-1, 2:end-1);

[Ny Nx Nz] = size(T);
T_min=-20;
T_max=20;

Lx= Nx;                  %  width (m)
Ly= Ny;                  %  depth (m)
Lz= Nz;                  %  height (m)
dx=Lx/Nx;               % spacing along x
dy=Ly/Ny;               % spacing along y
dz=Lz/Nz;               % spacing along z

numisosurf=25; % number of isosurfaces
isovalues=linspace(T_min,T_max,numisosurf);

x=linspace(0,Lx,Nx);
y=linspace(0,Ly,Ny);
z=linspace(0,Lz,Nz);
[X Y Z]= meshgrid(x,y,z);

figure('units','pixels','position',[100 100 1280 720])
plot3D(X,Y,Z,Lx,Ly,Lz,dx,dy,dz,T,T_min,T_max,isovalues,1,1); 

drawnow;
pause(10);