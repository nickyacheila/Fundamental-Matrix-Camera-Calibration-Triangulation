% This script 
% (1) Loads points and camera matrices
% (2) Performs triangulation to find 3d coordinates of points
% (3) Plots 3d points

clear
close all

% Load house 
camera_matrix_1 = load('data/house/house1_camera.txt');
camera_matrix_2 = load('data/house/house2_camera.txt');
matches = load('data/house/house_matches.txt');

