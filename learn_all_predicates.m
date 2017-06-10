high_mus = zeros(numfeats, 1, num_features_to_learn);
high_inv_sigmas = zeros(numfeats, numfeats, num_features_to_learn);
low_mus = zeros(numfeats, 1, num_features_to_learn);
low_inv_sigmas = zeros(numfeats, numfeats, num_features_to_learn);

for feat_to_learn = 1 : num_features_to_learn
    feat_name = feature_names{feat_to_learn};
    matfile = sprintf('%s%s_%s_low%d_%dpos_%dopp_%d_run%d.mat', pred_matfolder, input, ...
        feat_name, learn_lesser, num_train_pos, num_train_opp, run);
    
    % Try to learn high and low only if we haven't already learned them
    try
        load(matfile);
    catch
        % Get the appropriate animal vectors and names for this feature
        load(sprintf('data_animals_%s_%s.mat', input, feat_name));

        % Center dimensions for high-low data
        if use_centering
            high_animal_vecs = high_animal_vecs - repmat(dim_means, length(high_animal_names), 1);
            low_animal_vecs = low_animal_vecs - repmat(dim_means, length(low_animal_names), 1);
            med_animal_vecs = med_animal_vecs - repmat(dim_means, length(med_animal_names), 1);
        end

        if learn_lesser
            temp1 = high_animal_vecs;
            temp2 = high_animal_names;
            high_animal_vecs = low_animal_vecs;
            high_animal_names = low_animal_names;
            low_animal_vecs = temp1;
            low_animal_names = temp2;
        end;

        if strcmp(input, 'ratings')
            high_animal_vecs = high_animal_vecs(:, sel_dims);
            low_animal_vecs = low_animal_vecs(:, sel_dims);
            med_animal_vecs = med_animal_vecs(:, sel_dims); 
        elseif strcmp(input, 'topics')
            high_animal_vecs = high_animal_vecs(:, sel_dims) * scale_factor;
            low_animal_vecs = low_animal_vecs(:, sel_dims) * scale_factor;
            med_animal_vecs = med_animal_vecs(:, sel_dims) * scale_factor; 
        elseif strcmp(input, 'leuven')
            high_animal_vecs = high_animal_vecs(:, sel_dims) * scale_factor;
            low_animal_vecs = low_animal_vecs(:, sel_dims) * scale_factor;
            med_animal_vecs = med_animal_vecs(:, sel_dims) * scale_factor;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get training and test data for HIGH and LOW %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Animals that are high on this dimension
        num_high_animals = length(high_animal_names);
        train_pos_high_ind = randsample(num_high_animals, num_train_pos);
        train_pos_high_vecs = high_animal_vecs(train_pos_high_ind, :);
        train_pos_high_names = high_animal_names(train_pos_high_ind);

        train_opp_high_ind = randsample(train_pos_high_ind, num_train_opp);
        train_opp_high_vecs = high_animal_vecs(train_opp_high_ind, :);
        train_opp_high_names = high_animal_names(train_opp_high_ind);

        test_high_ind = setdiff(1:num_high_animals, train_pos_high_ind);
        test_high_vecs = high_animal_vecs(test_high_ind, :);
        test_high_names = high_animal_names(test_high_ind);

        % Animals that are low on this dimension
        num_low_animals = length(low_animal_names);
        train_pos_low_ind = randsample(num_low_animals, num_train_pos);
        train_pos_low_vecs = low_animal_vecs(train_pos_low_ind, :);
        train_pos_low_names = low_animal_names(train_pos_low_ind);

        train_opp_low_ind = randsample(train_pos_low_ind, num_train_opp);
        train_opp_low_vecs = low_animal_vecs(train_opp_low_ind, :);
        train_opp_low_names = low_animal_names(train_opp_low_ind);

        test_low_ind = setdiff(1:num_low_animals, train_pos_low_ind);
        test_low_vecs = low_animal_vecs(test_low_ind, :);
        test_low_names = low_animal_names(test_low_ind);

        % Animals that are in the middle on this dimension
        num_med_animals = length(med_animal_names);
        train_med_ind = randsample(num_med_animals, num_train_med);
        train_med_vecs = med_animal_vecs(train_med_ind, :);
        train_med_names = med_animal_names(train_med_ind);

        %%%%%%%%%%%%%%%%%%%%%%
        % Learn HIGH and LOW %
        %%%%%%%%%%%%%%%%%%%%%%

        mu_prior_pred = zeros(numfeats, 1);
        sigma_prior_pred = eye(numfeats);

        % Learn weights for HIGH
        traindata_high = [train_pos_high_vecs; train_opp_low_vecs; train_med_vecs];
        trainlabels = [ones(num_train_pos, 1); -1 * ones(num_train_opp + num_train_med, 1)];
        [mu_high, sigma_high, sigma_inv_high] = train_batch(traindata_high, ...
            trainlabels, mu_prior_pred, sigma_prior_pred, max_iter, criterion);

        % Learn weights for LOW
        traindata_low = [train_pos_low_vecs; train_opp_high_vecs; train_med_vecs];
        [mu_low, sigma_low, sigma_inv_low] = ...
            train_batch(traindata_low, trainlabels, mu_prior_pred, sigma_prior_pred, max_iter, criterion);
        
        % Save the learned matrices to file
        save(matfile, 'mu_high', 'mu_low', 'sigma_high', 'sigma_low', 'sigma_inv_high', 'sigma_inv_low');
    end

    % Save the learned matrices in memory
    high_mus(:, :, feat_to_learn) = mu_high;
    high_inv_sigmas(:, :, feat_to_learn) = sigma_inv_high;
    low_mus(:, :, feat_to_learn) = mu_low;
    low_inv_sigmas(:, :, feat_to_learn) = sigma_inv_low;
end