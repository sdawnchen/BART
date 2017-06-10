% This program tests all the analogy types after learning comparative relations.

clear all; 

input = 'ratings';%'topics';%'leuven';%
if strcmp(input, 'ratings')
    num_train_pairs_pos_vec = [1:10 20:20:100];
    num_train_opp = 0;
    num_runs = 100;
    numfeats = 4;
elseif strcmp(input, 'topics')
    num_train_pairs_pos_vec = [10:10:100 200 300];
    num_train_opp = 20;
    num_runs = 10;
    numfeats = 50;
elseif strcmp(input, 'leuven')
    num_train_pairs_pos_vec = [1:10 20:20:100];
    num_train_opp = 20;
    num_runs = 10;
    numfeats = 50;
end

BART = 1;
feature_names = {'size', 'fierceness', 'intelligence', 'speed'};
num_features_to_learn = 4;
num_pair_params = numfeats * 2;
num_train_pairs_neg_vec = 0;
run1 = 1;

% Parameters specific to analogy
learned_lesser = 1;     % 1: use learned lesser relation; 0: generate the lesser relation by reversing the greater relation
controlindx = 0;        % 0: keep everything (means, variances, and covariances); 1: keep variances, throw away covariances; 2: use identity matrix as cov matrix


if BART
    rel_matfolder = sprintf('results/%s/matrices/relations/BART/', input);
else
    rel_matfolder = sprintf('results/%s/matrices/relations/baseline/', input);
end
headings = {'# pos. train pairs', '# neg. train pairs', 'Seed #', 'Model acc'};

analogy_folder = sprintf('results/%s/analogy', input);
if ~exist(analogy_folder, 'dir')
    mkdir(analogy_folder)
end

for num_train_i = 1 : length(num_train_pairs_pos_vec)
    num_train_pairs_pos = num_train_pairs_pos_vec(num_train_i);
    num_train_pairs_neg = num_train_pairs_neg_vec;
   
    % Figure out prefix for results file name
    if BART
        filename_prefix = sprintf('results/%s/analogy/BART/%s_BART_train%dpos_%dneg_control%d', ...
            input, input, num_train_pairs_pos, num_train_pairs_neg, controlindx);
    else
        filename_prefix = sprintf('results/%s/analogy/baseline/%s_baseline_train%dpos_%dneg_control%d', ...
            input, input, num_train_pairs_pos, num_train_pairs_neg, controlindx);
    end
    analogy_results_file = [filename_prefix '.xls'];
     
    % Load all matrices
    allfeats_mu_higher = zeros(num_pair_params, 1, num_features_to_learn, num_runs);
    allfeats_sigma_higher = zeros(num_pair_params, num_pair_params, num_features_to_learn, num_runs);
    allfeats_mu_lower = zeros(num_pair_params, 1, num_features_to_learn, num_runs);
    allfeats_sigma_lower = zeros(num_pair_params, num_pair_params, num_features_to_learn, num_runs);
    
    for run = run1 : run1 + num_runs - 1
        runi = run - run1 + 1;
        
        for i = 1 : num_features_to_learn
            load(sprintf('%s%s_%s_lesser0_train%d_%d_run%d_mumat.mat', rel_matfolder, input, feature_names{i}, ...
                num_train_pairs_pos, num_train_pairs_neg, run));
            allfeats_mu_higher(:, :, i, runi) = mumat(:,2);
            
            if controlindx == 2
                sigma_post = eye(num_pair_params);
            else
                load(sprintf('%s%s_%s_lesser0_train%d_%d_run%d_sigma.mat', rel_matfolder, input, feature_names{i}, ...
                    num_train_pairs_pos, num_train_pairs_neg, run));
                if controlindx == 1
                    sigma_post = diag(diag(sigma_post));
                end
            end
            allfeats_sigma_higher(:, :, i, runi) = sigma_post;
            
            if learned_lesser
                load(sprintf('%s%s_%s_lesser1_train%d_%d_run%d_mumat.mat', rel_matfolder, input, feature_names{i}, ...
                num_train_pairs_pos, num_train_pairs_neg, run));
                allfeats_mu_lower(:, :, i, runi) = mumat(:,2);

                if controlindx == 2
                    sigma_post = eye(num_pair_params);
                else
                    load(sprintf('%s%s_%s_lesser1_train%d_%d_run%d_sigma.mat', rel_matfolder, input, feature_names{i}, ...
                        num_train_pairs_pos, num_train_pairs_neg, run));
                    if controlindx == 1
                        sigma_post = diag(diag(sigma_post));
                    end
                end
                allfeats_sigma_lower(:, :, i, runi) = sigma_post;
            else
                allfeats_mu_lower(:, :, i, runi) = reverse_roles_mu(allfeats_mu_higher(:, :, i, runi), numfeats);
                allfeats_sigma_lower(:, :, i, runi) = reverse_roles_sigma(allfeats_sigma_higher(:, :, i, runi), numfeats);
            end
        end
    end
    
    % Run tests
    fprintf('Testing Same-O analogies: same extreme, opposite as foil');
    analogy_test_SameO;
    
    fprintf('Testing Same-OE analogies: same extreme, opposite extreme as foil');
    analogy_test_SameOE;
    
    fprintf('Testing O-S analogies: opposite, split pair as foil');
    analogy_test_OppS;
    
    fprintf('Testing O-R analogies: opposite, reversed as foil');
    analogy_test_OppR;
    
    fprintf('Testing O-C analogies: opposite, conflict foil');
    analogy_test_OppC;
end
