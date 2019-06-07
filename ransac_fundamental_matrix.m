function [F_matrix, matched_points_a, matched_points_b] = ransac_fundamental_matrix(matches1, matches2)
s=8;
d_threshold=1;
T= size(matches1,1)*70/100;
max_inliers=0;
best_fund_matrix=zeros(3,3);

for i =1:1000
    distances=-ones(size(matches1,1),1);
     s_ind=randperm(size(matches1,1),s);
     fund_matrix=estimate_fund_matrix([matches1(s_ind,:) matches2(s_ind,:)]);
     
     for j=1:size(matches1,1)
        distances(j)=abs([matches1(j,:) 1]*fund_matrix*[matches2(j,:) 1]');
     end
     inliers_ind=distances(distances<d_threshold);
     
     inliers_size=size(inliers,1);
     
     if inliers_size>T
         if inliers_size>max_inliers
             max_inliers=inliers_size;
             best_fund_matrix=fund_matrix;
         end
     end

end
%fit

end

