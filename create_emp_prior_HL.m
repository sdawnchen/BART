% Use the training data to select a single-place predicate to construct
% the empirical prior for learning comparative relations.

num_predicates = num_features_to_learn * 2;     % 2 predicates for each feature
predicate_votes = zeros(1, num_predicates);

% Go through all positive training examples, collecting votes for which
% predicate is most likely to apply to role 1
for i = 1 : size(train_pos_vecs, 1)
    animal1 = train_pos_vecs(i, 1 : numfeats)';
    animal2 = train_pos_vecs(i, numfeats + 1 : end)';

    % Go through all high predicates
    for j = 1 : num_predicates
        if j<=num_features_to_learn
            muvec = high_mus(:, :, j); 
            sigma_inv_mat = high_inv_sigmas(:, :, j);
        elseif j<=num_features_to_learn*2
            muvec = low_mus(:, :, j-num_features_to_learn);
            sigma_inv_mat = low_inv_sigmas(:, :, j-num_features_to_learn);
        end;    
                
        % Calculate the probability of this predicate applying to each
        % animal in the training pair
        animal1prob = compute_score(animal1, 1, muvec, sigma_inv_mat, max_iter, criterion); 
        animal2prob = compute_score(animal2, 1, muvec, sigma_inv_mat, max_iter, criterion); 

        % Increase the vote for the high predicate (for role 1) if it is
        % more likely to apply to animal1 than to animal2
        if animal1prob > animal2prob
            predicate_votes(j) = predicate_votes(j) + 1;
        end
    end
end

% Create weights from the winning predicate(s)
best_predicates = find(predicate_votes == max(predicate_votes));

% If there's a tie, average the weights of the tied predicates
if length(best_predicates) > 1
    mu_role1 = zeros(numfeats, 1);
    mu_role2 = zeros(numfeats, 1);
    for pind = 1:length(best_predicates)
        if best_predicates(pind) <= num_features_to_learn     % this winning predicate is a high predicate
            % Add that high predicate for role 1 and subtract it for role 2
            mu_role1 = mu_role1 + high_mus(:, :, best_predicates(pind));
            mu_role2 = mu_role2 - high_mus(:, :, best_predicates(pind)); 
        else    % this winning predicate is a low predicate
            mu_role1 = mu_role1 + low_mus(:, :, best_predicates(pind) - num_features_to_learn); 
            mu_role2 = mu_role2 - low_mus(:, :, best_predicates(pind) - num_features_to_learn);
        end       
    end
    
    % Divide by the number of winning predicates
    mu_role1 = mu_role1/length(best_predicates);
    mu_role2 = mu_role2/length(best_predicates);
 
else
    if best_predicates <= num_features_to_learn     % the winning predicate is a high predicate
        % Use that high predicate for role 1 and reverse signs for role 2
        mu_role1 = high_mus(:, :, best_predicates);
        mu_role2 = -mu_role1; 
    else    % the winning predicate is a low predicate
        mu_role1 = low_mus(:, :, best_predicates);
        mu_role2 = -mu_role1; 
    end
end

% Update mu_prior
mu_prior(1 : numfeats) = mu_role1;
mu_prior(numfeats + 1 : end) = mu_role2;

