% This script 
% (1) Loads points and camera matrices
% (2) Performs triangulation to find 3d coordinates of points
% (3) Plots 3d points

clear
close all

% Load house 
% camera_matrix_1 = load('data/house/house1_camera.txt');
% camera_matrix_2 = load('data/house/house2_camera.txt');
% matches = load('data/house/house_matches.txt');

% Load Library image
camera_matrix_1 = load('data/library/library1_camera.txt');
camera_matrix_2 = load('data/library/library2_camera.txt');
matches = load('data/library/library_matches.txt');

% Initiate empty matrix to save the 3D coordinates of matching points
coordinates=zeros(size(matches));

% Calculate camera centers in 3D
[U1,S1,V1]=svd(camera_matrix_1);
camera_1_coordinates=V1(:, length(V1))/V1(end);
[U2,S2,V2]=svd(camera_matrix_2);
camera_2_coordinates=V2(:, length(V2))/V2(end);

figure(1)
% Iterate over matching points 
for i=1:size(matches, 1)
   A = zeros(4);
   
   % Get u,v,u',v' values
   u = matches(i, 1);
   v = matches(i, 2);
   u_prime = matches(i, 3);
   v_prime = matches(i, 4);
   
   % Calculate A Matrix
   A(1, :) = v * camera_matrix_1(3, :) - camera_matrix_1(2, :);
   A(2, :) = u * camera_matrix_1(3, :) - camera_matrix_1(1, :);
   A(3, :) = v_prime * camera_matrix_2(3, :) - camera_matrix_2(2, :);
   A(4, :) = u_prime * camera_matrix_2(3, :) - camera_matrix_2(1, :);
   
   % Perform singular value decomposition to find 3D coordinates of point
   [U,S,V]=svd(A);
   X=V(:, length(V))/V(end);
   
   % Add to coordinates matrix
   coordinates(i, :)=X;
   
   % Plot lines starting from camera 1 and camera 2 center and going to
   % this point. Therefore plot two lines for each point that we have.
   % Notice that the Z axis is the axis passing through the camera lenses
   % and Y axis is the one that goes up.
   plot3([camera_1_coordinates(1), X(1)],[camera_1_coordinates(3), X(3)],[camera_1_coordinates(2), X(2)])
   hold on
   plot3([camera_2_coordinates(1), X(1)],[camera_2_coordinates(3), X(3)],[camera_2_coordinates(2), X(2)])
   hold on

end
xlabel('x')
ylabel('z')
zlabel('y')
hold off

% Plot coordinates as points on the 3D world. Again Z and Y are inverted. 
figure(2)
scatter3(camera_1_coordinates(1), camera_1_coordinates(3), camera_1_coordinates(2))
hold on
scatter3(camera_2_coordinates(1), camera_2_coordinates(3), camera_2_coordinates(2))
hold on
scatter3(coordinates(:, 1), coordinates(:, 3), coordinates(:, 2))
hold off

