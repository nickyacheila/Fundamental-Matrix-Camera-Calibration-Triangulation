function [best_fund_matrix, matched_points_a, matched_points_b] = ransac_fundamental_matrix(matches1, matches2)
s=8;
d_threshold=0.8;
T= 80;
max_inliers=0;
best_indexes=zeros(size(matches1,1),1);
best_distances=zeros(size(matches1,1),1);

for i =1:1000
    % Initiate distances to -1
    distances = Inf(size(matches1,1),1);

    % Get the indices of the sample points
    s_indices=randperm(size(matches1,1),s);
    
    % Calculate the fundamental matrix from the sampled points
    fund_matrix=estimate_fund_matrix([matches1(s_indices,:) matches2(s_indices,:)]);
    
    % Iterate over matching points
    for j=1:size(matches1,1)
        % Calculate distance of point from model
        distance = abs([matches1(j,:) 1]*fund_matrix*[matches2(j,:) 1]');
        
        % Update distance array only if the threshold is met
         if distance < d_threshold
             distances(j)=distance;
         end
    end
    
    % get number of inliers for this model
    inliers_size=nnz(distances<Inf);
    
    % if the inliers are more than threshold and more than the max_inliers
    % update the max values of fundamental matrix and inliers found
    if inliers_size>T && inliers_size>max_inliers
        max_inliers=inliers_size;
        best_indexes=(distances<Inf);
        best_distances=distances;
    end

end


matched_points_a=matches1(best_indexes,:);
matched_points_b=matches2(best_indexes,:);

best_fund_matrix=estimate_fund_matrix([matched_points_a matched_points_b]);

disp(max_inliers)

end

