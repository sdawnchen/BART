% Set up results files
analogy_row = 2;
sheet = 'SameO';
xlswrite(analogy_results_file, headings, sheet);

analogy_data_file = [filename_prefix '_SameO_data.txt'];
datafile = fopen(analogy_data_file, 'w');

for run = run1 : run1 + num_runs - 1
    runi = run - run1 + 1;
    fprintf('\nRun %d\n\n', run);
    fprintf(datafile, '\nRun %d\n\n', run);
    
    num_correct = 0;
    num_tests = 0;

    % feat1 (for A)
    for feat1 = 1 : num_features_to_learn
        feat1_higher_mu = allfeats_mu_higher(:, :, feat1, runi);
        feat1_higher_sigma = allfeats_sigma_higher(:, :, feat1, runi);
        feat1_lower_mu = allfeats_mu_lower(:, :, feat1, runi);
        feat1_lower_sigma = allfeats_sigma_lower(:, :, feat1, runi);
        
        % feat2 (for B) can be anything but feat1
        for feat2 = setdiff(1 : num_features_to_learn, feat1)
            feat2_higher_mu = allfeats_mu_higher(:, :, feat2, runi);
            feat2_higher_sigma = allfeats_sigma_higher(:, :, feat2, runi);
            feat2_lower_mu = allfeats_mu_lower(:, :, feat2, runi);
            feat2_lower_sigma = allfeats_sigma_lower(:, :, feat2, runi);
            
            % feat3 (for C) and feat4 (for D) are the remaining features
            remaining = setdiff(1 : num_features_to_learn, [feat1 feat2]);
            feat3 = remaining(1);
            feat4 = remaining(2);
            
            feat3_higher_mu = allfeats_mu_higher(:, :, feat3, runi);
            feat3_higher_sigma = allfeats_sigma_higher(:, :, feat3, runi);
            feat3_lower_mu = allfeats_mu_lower(:, :, feat3, runi);
            feat3_lower_sigma = allfeats_sigma_lower(:, :, feat3, runi);
            
            feat4_higher_mu = allfeats_mu_higher(:, :, feat4, runi);
            feat4_higher_sigma = allfeats_sigma_higher(:, :, feat4, runi);
            feat4_lower_mu = allfeats_mu_lower(:, :, feat4, runi);
            feat4_lower_sigma = allfeats_sigma_lower(:, :, feat4, runi);
            
            % Do tests for higher relations as the true analogy
            [true_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_higher_mu, feat2_higher_mu, ...
                feat3_higher_mu, feat4_higher_mu, feat1_higher_sigma, feat2_higher_sigma, feat3_higher_sigma, feat4_higher_sigma);
            
            fprintf('%s : %s :: %s : %s, dist = %f, vs.\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat3, true), get_relation_name(feat4, true), true_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, vs.\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat3, true), get_relation_name(feat4, true), true_dist);
            
            
            [false_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_higher_mu, feat2_higher_mu, ...
                feat3_higher_mu, feat3_lower_mu, feat1_higher_sigma, feat2_higher_sigma, feat3_higher_sigma, feat3_lower_sigma);
            
            if true_dist < false_dist
                num_correct = num_correct + 1;
            end
            num_tests = num_tests + 1;
            
            fprintf('%s : %s :: %s : %s, dist = %f\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat3, true), get_relation_name(feat3, false), false_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, %d\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat3, true), get_relation_name(feat3, false), false_dist, true_dist < false_dist);
            
            
            [false_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_higher_mu, feat2_higher_mu, ...
                feat4_lower_mu, feat4_higher_mu, feat1_higher_sigma, feat2_higher_sigma, feat4_lower_sigma, feat4_higher_sigma);
            
            if true_dist < false_dist
                num_correct = num_correct + 1;
            end
            num_tests = num_tests + 1;
            
            fprintf('%s : %s :: %s : %s, dist = %f\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat4, false), get_relation_name(feat4, true), false_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, %d\n', ...
                get_relation_name(feat1, true), get_relation_name(feat2, true), ...
                get_relation_name(feat4, false), get_relation_name(feat4, true), false_dist, true_dist < false_dist);
            fprintf('\n'); fprintf(datafile,'\n');
            
            
            % Do analogy tests for lower relations as the true analogy
            [true_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_lower_mu, feat2_lower_mu, ...
                feat3_lower_mu, feat4_lower_mu, feat1_lower_sigma, feat2_lower_sigma, feat3_lower_sigma, feat4_lower_sigma);
            
            fprintf('%s : %s :: %s : %s, dist = %f, vs.\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat3, false), get_relation_name(feat4, false), true_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, vs.\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat3, false), get_relation_name(feat4, false), true_dist);
            
            
            [false_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_lower_mu, feat2_lower_mu, ...
                feat3_lower_mu, feat3_higher_mu, feat1_lower_sigma, feat2_lower_sigma, feat3_lower_sigma, feat3_higher_sigma);
            
            if true_dist < false_dist
                num_correct = num_correct + 1;
            end
            num_tests = num_tests + 1;
            
            fprintf('%s : %s :: %s : %s, dist = %f\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat3, false), get_relation_name(feat3, true), false_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, %d\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat3, false), get_relation_name(feat3, true), false_dist, true_dist < false_dist);
            
            
            [false_dist, ABsortorder, CDsortorder, AB_murst, CD_murst] = ...
                analogy_importance_mapping(feat1_lower_mu, feat2_lower_mu, ...
                feat4_higher_mu, feat4_lower_mu, feat1_lower_sigma, feat2_lower_sigma, feat4_higher_sigma, feat4_lower_sigma);
            
            if true_dist < false_dist
                num_correct = num_correct + 1;
            end
            num_tests = num_tests + 1;
            
            fprintf('%s : %s :: %s : %s, dist = %f\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat4, true), get_relation_name(feat4, false), false_dist);
            
            fprintf(datafile, '%s : %s :: %s : %s, dist = %f, %d\n', ...
                get_relation_name(feat1, false), get_relation_name(feat2, false), ...
                get_relation_name(feat4, true), get_relation_name(feat4, false), false_dist, true_dist < false_dist);
            
            fprintf('\n'); fprintf(datafile,'\n');
        end
    end
    
    %         num_tests = num_features_to_learn * (num_features_to_learn - 1)^2;
    accuracy = num_correct / num_tests;
    xlswrite(analogy_results_file, [num_train_pairs_pos, ...
        num_train_pairs_neg, run, accuracy], sheet, sprintf('A%d', analogy_row));
    analogy_row = analogy_row + 1;
    fprintf('accuracy = %f\n',accuracy);
end
fclose all;
