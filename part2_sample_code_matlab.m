% This script 
% (1) Loads and resizes pair images
% (2) Calls VLFeat's SIFT matching functions
% (3) Estimates the fundamental matrix using RANSAC 
%     and filters away spurious matches (you code this)
% (4) Draws the epipolar lines on images and corresponding matches

clear
close all

% The Notre Dame pair is difficult because the keypoints are largely on the
% same plane. Still, even an inaccurate fundamental matrix can do a pretty
% good job of filtering spurious matches.
I1 = imread('data/NotreDame/NotreDame1.jpg');
I2 = imread('data/NotreDame/NotreDame2.jpg');
I1 = imresize(I1, 0.5, 'bilinear');
I2 = imresize(I2, 0.5, 'bilinear');


% Finds matching points in the two images using VLFeat's implementation of
% SIFT There can still be many spurious matches, though. ( check the
% plots!!!
%% Important note: you might need to change the parameters inside this function to have more or better matches
[matches1, matches2] = find_matching_points( I1, I2 );
fprintf('Found %d possibly matching features\n',size(matches1,1));
plot_correspondence(I1, I2, matches1(:,1), matches1(:,2), matches2(:,1),matches2(:,2));
pause;

%% Calculate the fundamental matrix using RANSAC
[F_matrix, matched_points_a, matched_points_b] = ransac_fundamental_matrix(matches1, matches2);

%% plot the best matched points following RANSAC for the fundamental matrix
plot_correspondence(I1, I2, matched_points_a(:,1), matched_points_a(:,2), matched_points_b(:,1),matched_points_b(:,2));
pause;

%% Draw the epipolar lines on the images and corresponding matches
draw_epipolar_lines(F_matrix, I1, I2, matched_points_a, matched_points_b);

% We are already reestimating inside the ransac_fundamental_matrix function
% %optional - re estimate the fundamental matrix using ALL the inliers.
% [ F_matrix ] = estimate_fundamental_matrix_james(matched_points_a, matched_points_b);
% draw_epipolar_lines(F_matrix, I1, I2, matched_points_a, matched_points_b);