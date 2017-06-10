function KLdist = kldistnormmatrix(mu1,cov1,mu2,cov2)

% mus: n by 1, covs: n by n
KLdist = 0.5*(mu1-mu2)'*(inv(cov1)+inv(cov2))*(mu1-mu2) + ...
    0.5*trace(inv(cov1)*cov2+inv(cov2)*cov1-2*eye(size(cov1)) );


% (AB_mu-CD_mu)'*(inv(AB_cov))*(AB_mu-CD_mu)

% temp1 = (Amu_sort(1:i)-Cmu_sort(1:i))'*(inv(covA_sort(1:i,1:i)))*(Amu_sort(1:i)-Cmu_sort(1:i))...
%     +(Bmu_sort(1:i)-Dmu_sort(1:i))'*(inv(covB_sort(1:i,1:i)))*(Bmu_sort(1:i)-Dmu_sort(1:i))
