clear; close all; clc;

filename = 'vector_data.csv';
dimFileName = 'dim.csv';
info = readtable(dimFileName,'Delimiter', ',', 'HeaderLines', 0);
T = reshape(readmatrix(filename), info{1,1:3});
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

p = figure('units','pixels','position',[100 100 1280 720]);
plot3D(X,Y,Z,Lx,Ly,Lz,dx,dy,dz,T,T_min,T_max,isovalues,char(info{1,4}), info{1,5},info{1,6}); 
uiwait(p)
