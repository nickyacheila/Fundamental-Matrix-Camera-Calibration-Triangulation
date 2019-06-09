function [F_matrix] = estimate_fund_matrix(matches)
persistent n;
if isempty(n)
        n = 0;
end
% create matrix A
A=ones(size(matches,1),9);
A(:,1)=matches(:,1).*matches(:,3);
A(:,2)=matches(:,1).*matches(:,4);
A(:,3)=matches(:,1);
A(:,4)=matches(:,3).*matches(:,2);
A(:,5)=matches(:,2).*matches(:,4);
A(:,6)=matches(:,2);
A(:,7)=matches(:,3);
A(:,8)=matches(:,4);

%first SVD
[~,S,V]=svd(A);
%find min singular value
[~,I]=min(diag(S));
f=V(:,I);

F_matrix1=reshape(f,3,3);

[U1,S1,V1]=svd(F_matrix1);
[~,I1]=min(diag(S1));

S1(I1,I1)=0;

F_matrix=U1*S1*V1';
f1=reshape(F_matrix,9,1);
Z=A*f1;
if sum(Z<1)<2
    n=n+1;
%     disp(n);
end

end

