function [finaldist, AB_sort_order, CD_sort_order, mapped_AB_mu, mapped_CD_mu, mapped_AB_cov, mapped_CD_cov] = ...
    analogy_importance_mapping(muA, muB, muC, muD, covA, covB, covC, covD)

numfeats = size(muA, 1)/2;

% Calculate the importance of dimensions for AB pairs
importance_dims_AB = zeros(numfeats, 1);
for i = 1:numfeats
    % Get the sum of each weight (divided by its SD) over the two roles and
    % two relations (A and B)
    importance_dimi_A = abs(muA(i)) / sqrt(covA(i, i)) + ...
        abs(muA(numfeats + i)) / sqrt(covA(numfeats + i, numfeats + i));
    importance_dimi_B = abs(muB(i)) / sqrt(covB(i, i)) + ...
        abs(muB(numfeats + i)) / sqrt(covB(numfeats + i, numfeats + i));
    importance_dims_AB(i) = importance_dimi_A + importance_dimi_B;
end

% Sort the feature dimensions based on importance
[importance_sorted_AB, AB_sort_order_half] = sort(importance_dims_AB, 'descend');

% Get the sorted feature dimensions for both roles. Note that each
% dimension in role 1 is immediately followed by its corresponding
% dimension in role 2.
AB_sort_order = [];
for i = 1:numfeats
    AB_sort_order = [AB_sort_order, AB_sort_order_half(i), AB_sort_order_half(i) + numfeats];
end

% Sort the A and B mu vectors
muA_sorted = muA(AB_sort_order);
muB_sorted = muB(AB_sort_order);

% Sort the A and B cov matrices
covA_sorted = covA(AB_sort_order, AB_sort_order);
covB_sorted = covB(AB_sort_order, AB_sort_order);

% Map dimensions of CD to dimensions of AB
% For each dimension in AB, find the dimension in CD that has the
% smallest KL-distance (in terms of its mean weights and cov matrix)
dim_chosen = false(numfeats,1);
CD_sort_order_half = zeros(numfeats,1);
count = 0;
% Go through dimensions in AB
for i = 1:2:size(muA,1)
    count = count+1;
    % Form AB matrices needed for calculating KL-distance
    pdistmu = [muA_sorted(i); muA_sorted(i+1); muB_sorted(i); muB_sorted(i+1)];
    pdistcov = [covA_sorted(i,i) covA_sorted(i,i+1) 0 0;
        covA_sorted(i+1,i) covA_sorted(i+1,i+1) 0 0;
        0 0 covB_sorted(i,i) covB_sorted(i,i+1);
        0 0 covB_sorted(i+1,i) covB_sorted(i+1,i+1)];
    kldist1 = zeros(numfeats,1);
    % Go through dimensions in CD
    for j = 1:numfeats
        if ~dim_chosen(j)   % If this dimension hasn't already been chosen by a different dimension in AB
            % Form CD matrices needed for calculating KL-distance
            qdistmu = [muC(j); muC(numfeats+j); muD(j); muD(numfeats+j)];
            qdistcov = [covC(j,j) covC(j, numfeats+j) 0 0;
                covC(numfeats+j, j) covC(numfeats+j, numfeats+j) 0 0;
                0 0 covD(j,j) covD(j, numfeats+j);
                0 0 covD(numfeats+j, j) covD(numfeats+j, numfeats+j)];
            kldist1(j) = kldistnormmatrix(pdistmu, pdistcov, qdistmu, qdistcov);
        else
            kldist1(j) = inf;
        end
    end
    [temp, min_dist_dim] = min(kldist1);
    CD_sort_order_half(count) = min_dist_dim(1);
    dim_chosen(min_dist_dim(1)) = true; 
end

CD_sort_order = [];
for i = 1:numfeats
    CD_sort_order = [CD_sort_order, CD_sort_order_half(i), CD_sort_order_half(i) + numfeats];
end

% Sort the C and D mu vectors based on mapping to AB
muC_sorted = muC(CD_sort_order);
muD_sorted = muD(CD_sort_order);

% Sort the C and D cov matrices based on mapping to AB
covC_sorted = covC(CD_sort_order, CD_sort_order);
covD_sorted = covD(CD_sort_order, CD_sort_order);

% Calculate KL-distance between mapped AB and CD
mapped_AB_mu = [muA_sorted; muB_sorted];
mapped_AB_cov = [covA_sorted zeros(size(covA_sorted)); zeros(size(covA_sorted)) covB_sorted];

mapped_CD_mu = [muC_sorted; muD_sorted];
mapped_CD_cov = [covC_sorted zeros(size(covC_sorted)); zeros(size(covC_sorted)) covD_sorted];

finaldist = kldistnormmatrix(mapped_AB_mu, mapped_AB_cov, mapped_CD_mu, mapped_CD_cov);

