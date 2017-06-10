% This is the main program to train and test the model on comparative relations.
% The (optional) input arguments are:
% input - input source, either 'ratings', 'leuven', or 'topics'
% BART - model flag, set to either 0 or 1. 0: baseline model, 1: BART
% learn_lesser - relation flag, set to either 0 or 1. 0: learn the
% "greater" relations, 1: learn the "lesser" relations
% num_train_pairs_pos - the number of postive training examples

% The function computes the posterior probability distribution p(w|X,R)~
% N(mu_post, sigma_post). The prior, p(w), is determined by the prior
% selection procedure for BART and p(w)~N(0,I) for the baseline model.

% The generalization results are saved in the
% results/<input>/generalization folder, and the resulting relational
% representations are saved in the results/<input>/matrices folder (see
% the DIRECTORY STRUCTURE section in README.md for more details).


function [] = learn_test_relations(input, BART, learn_lesser, num_train_pairs_pos)
tic;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 1
    input = 'ratings';          % input source, either 'ratings', 'leuven', or 'topics'
end
if nargin < 2
    BART = 1;                   % 0: baseline model, 1: BART
end
if nargin < 3
    learn_lesser = 0;           % 0: learn greater relations, 1: learn lesser relations
end

% Input-specific parameters
if strcmp(input,'ratings')    
    num_sel_dims = 0;           % number of dimensions to select, or 0 if shouldn't select dimensions
    num_train_pos = 20;         % number of positive training examples for learning single-place predicates
    num_train_opp = 0;          % number of opposite-extreme negative training examples for predicates
    num_train_med = 0;          % number of medium negative training examples for predicates
    if nargin < 4
        num_train_pairs_pos = 20;	% number of positive training examples for comparative relations
    end
    num_train_pairs_neg = 0;	% number of negative training examples for relations
    use_hyperprior = false;     % whether to use hyperprior
    num_runs = 100;             % total number of runs
else    % for Leuven and topics inputs
    num_sel_dims = 50;          % number of dimensions to select, or 0 if shouldn't select dimensions
    num_train_pos = 20;         % number of positive training examples for learning single-place predicates
    num_train_opp = 20;         % number of opposite-extreme negative training examples for predicates
    num_train_med = 0;          % number of medium negative training examples for predicates
    if nargin < 4
        num_train_pairs_pos = 100;	% number of positive training examples for comparative relations
    end
    num_train_pairs_neg = 0;    % number of negative training examples for relations
    use_hyperprior = true;      % whether to use hyperprior
    num_runs = 10;              % total number of runs
    scale_factor = 100;         % factor to scale the inputs
end

% Less commonly changed parameters
feature_names = {'size', 'fierceness', 'intelligence', 'speed'};
num_features_to_learn = 4;
min_feat_diff = 0.5;            % minimum difference between two animals on the feature of interest
zero_prior_variance = 1;        % variance of zero prior (for baseline model)
use_centering = true;           % whether to center the input data to have a mean of about 0
run1 = 1;                      % seed for run 1  

% For variational method
max_iter = 5000;                % maximum number of iterations
criterion = 0.00001;            % criterion for convergence

% Hyperprior parameters (b0/a0 is mean variance)
a0 = 5;
b0 = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up info for writing results to file %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if BART
    results_folder = sprintf('results/%s/generalization/BART', input);
    results_file = sprintf('%s/%s_BART_lesser%d_train%dpos_%dneg.xls', ...
        results_folder, input, learn_lesser, num_train_pairs_pos, num_train_pairs_neg);
    rel_matfolder = sprintf('results/%s/matrices/relations/BART/', input);
else
    results_folder = sprintf('results/%s/generalization/baseline', input);
    results_file = sprintf('%s/%s_baseline_lesser%d_train%dpos_%dneg.xls', ...
        results_folder, input, learn_lesser, num_train_pairs_pos, num_train_pairs_neg);
    rel_matfolder = sprintf('results/%s/matrices/relations/baseline/', input);
end
pred_matfolder = sprintf('results/%s/matrices/predicates/', input);

% Create all the necessary directories
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end
    
if ~exist(rel_matfolder, 'dir')
    mkdir(rel_matfolder);
end

if ~exist(pred_matfolder, 'dir')
    mkdir(pred_matfolder);
end

headings = {'Seed #', 'a0', 'b0', 'Final iter', '# pairs tested', 'Abs acc', 'Rel acc', 'Az'};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get input data and do pre-processing %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get the appropriate animal vectors and names
load('data_animals_ratings_all.mat');
if strcmp(input, 'topics')
    load('data_animals_topics_all.mat');
    numfeats_orig = size(all_animal_topic_vecs, 2);
elseif strcmp(input, 'leuven')
    load('data_animals_leuven_all.mat');
    numfeats_orig = size(all_animal_leuven_vecs, 2);
    
    % Use the subset of animals in the Leuven dataset
    all_animal_rating_vecs = all_animal_rating_vecs(all_leuven_indices, :);
    all_animal_names = all_animal_names(all_leuven_indices);
else
    numfeats_orig = size(all_animal_rating_vecs, 2);
end

% Figure out some basic info
if num_sel_dims == 0
    numfeats = numfeats_orig;
else
    numfeats = num_sel_dims;
end
num_pair_feats = numfeats * 2;
num_animals = length(all_animal_names);

% Select dimensions
if num_sel_dims > 0
    if strcmp(input, 'topics')
        [temp, sorted_dims] = sort(sum(all_animal_topic_vecs), 'descend');
    elseif strcmp(input, 'leuven')
        [temp, sorted_dims] = sort(sum(all_animal_leuven_vecs), 'descend');
        % Delete dimension 441, 'is small'
        sorted_dims(sorted_dims == 441) = [];
    end
    sel_dims = sorted_dims(1 : num_sel_dims);
else
    sel_dims = 1:numfeats;
end
save(sprintf('%s%s_seldims.mat', rel_matfolder, input), 'sel_dims');

% If desired, center each dimension separately
if use_centering
    if strcmp(input, 'topics')
        load('data_topics_dim_means.mat');
        all_animal_topic_vecs = all_animal_topic_vecs - repmat(dim_means, num_animals, 1);
    elseif strcmp(input, 'leuven')
        load('data_leuven_dim_means.mat');
        all_animal_leuven_vecs = all_animal_leuven_vecs - repmat(dim_means, num_animals, 1);
    else
        load('data_ratings_dim_means.mat');
        all_animal_rating_vecs = all_animal_rating_vecs - repmat(dim_means, num_animals, 1);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%
% Run the simulations %
%%%%%%%%%%%%%%%%%%%%%%%

for feature_to_learn = 1 : num_features_to_learn
    feature_name = feature_names{feature_to_learn};
    fprintf('Learning about %s\n', feature_name);
    xlswrite(results_file, headings, feature_name, 'A1');
    
    % Sort the animals based on the feature we're interested in
    [temp, sortedi] = sort(all_animal_rating_vecs(:, feature_to_learn), 'descend');  
    sorted_animal_rating_vecs = all_animal_rating_vecs(sortedi, :);
    sorted_animal_names = all_animal_names(sortedi, :);
    
    if strcmp(input, 'ratings')
        sorted_animal_vecs = all_animal_rating_vecs(sortedi, sel_dims); 
    elseif strcmp(input, 'topics')
        sorted_animal_vecs = all_animal_topic_vecs(sortedi, sel_dims) * scale_factor;
    elseif strcmp(input, 'leuven')
        sorted_animal_vecs = all_animal_leuven_vecs(sortedi, sel_dims) * scale_factor;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create the entire pool of possible training and test items %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Create all pairs of animals that instantiate the relation
    tot_num_pos_pairs = num_animals * (num_animals - 1) / 2;
    all_pos_vecs = zeros(tot_num_pos_pairs, numfeats * 2);
    all_pos_names = cell(tot_num_pos_pairs, 2);
    start_row = 1;
    for i = 1 : num_animals - 1
        numpairs = num_animals - i;
        if learn_lesser == 0
            all_pos_vecs(start_row : start_row + numpairs - 1, :) = ...
                [repmat(sorted_animal_vecs(i, :), numpairs, 1), sorted_animal_vecs(i+1 : num_animals, :)];
            all_pos_names(start_row : start_row + numpairs - 1, :) = ...
                [repmat(sorted_animal_names(i, :), numpairs, 1), sorted_animal_names(i+1 : num_animals, :)];
        else
            all_pos_vecs(start_row : start_row + numpairs - 1, :) = ...
                [sorted_animal_vecs(i+1 : num_animals, :), repmat(sorted_animal_vecs(i, :), numpairs, 1)];
            all_pos_names(start_row : start_row + numpairs - 1, :) = ...
                [sorted_animal_names(i+1 : num_animals, :), repmat(sorted_animal_names(i, :), numpairs, 1)];
        end
        start_row = start_row + numpairs;
    end
    
    % Remove pairs of animals that do not satisfy the minimum difference
    % criterion
    to_remove = [];
    start = 0;
    for i = 1 : num_animals - 1
        j = i + 1;
        featval1 = sorted_animal_rating_vecs(i, feature_to_learn);
        featval2 = sorted_animal_rating_vecs(j, feature_to_learn);
        % Go down the list for the second animal until it differs by a
        % sufficient amount
        while abs(featval1 - featval2) < min_feat_diff
            % Add the index for this pair to to_remove
            to_remove = [to_remove, start + (j - i)];
            j = j + 1;
            if j <= num_animals
                featval2 = sorted_animal_rating_vecs(j, feature_to_learn);
            else
                break;
            end
        end
        % Update the starting index
        start = start + num_animals - i;
    end

    all_pos_vecs(to_remove, :) = [];
    all_pos_names(to_remove, :) = [];
    tot_num_pos_pairs = size(all_pos_vecs, 1);

    row = 2;   % for writing to Excel file   
    for run = run1 : run1 + num_runs - 1
        fprintf('Run %d\n', run);
        rand('seed', learn_lesser*100 + run);   % set random seed
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Randomly select training data for this run %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Select positive training examples
        train_pos_inds = randsample(tot_num_pos_pairs, num_train_pairs_pos);
        train_pos_vecs = all_pos_vecs(train_pos_inds, :);
        train_pos_names = all_pos_names(train_pos_inds, :);

        % Select negative training examples
        train_neg_inds = randsample(tot_num_pos_pairs, num_train_pairs_neg);
        train_neg_vecs = all_pos_vecs(train_neg_inds, [numfeats + 1 : end, 1 : numfeats]);
        train_neg_names = all_pos_names(train_neg_inds, [2 1]);
        
        % Combine the training examples
        all_train_data = [train_pos_vecs; train_neg_vecs];
        all_train_labels = [ones(num_train_pairs_pos, 1); -1 * ones(num_train_pairs_neg, 1)];
 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Use the remaining pairs as test data %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Test all possible pairs, excluding any pairs (or their reverses)
        % that were used in the training set
        test_pos_inds = setdiff(1:tot_num_pos_pairs, union(train_pos_inds, train_neg_inds));
        test_pos_vecs = all_pos_vecs(test_pos_inds, :);
        test_pos_names = all_pos_names(test_pos_inds, :);
        num_test_pairs = length(test_pos_inds);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Obtain prior distribution of weights %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Create uninformative prior
        mu_prior = zeros(num_pair_feats, 1);
        sigma_prior = eye(num_pair_feats) * zero_prior_variance;
        
        % Create empirical prior if using BART model
        if BART   
            learn_all_predicates;
            create_emp_prior;
        end

        
        %%%%%%%%%%%%%%%%%%%%%%
        % Learn the relation %
        %%%%%%%%%%%%%%%%%%%%%%
        
        if use_hyperprior
            [mu_post, sigma_post, sigma_inv_post, logdetV, E_a, L, predprob, ...
                final_iter] = bayes_logit_fit_nonzero_mean(all_train_data, ...
                all_train_labels, mu_prior, a0, b0, max_iter, criterion);
        else
            [mu_post, sigma_post, sigma_inv_post] = train_batch(all_train_data, ...
                all_train_labels, mu_prior, sigma_prior, max_iter, criterion);
            a0 = NaN;
            b0 = NaN;
            final_iter = NaN;
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Evaluate performance and save results %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        evaluate_relation;

        % Print out results
        fprintf('Absolute accuracy: %f\n', abs_acc_model);
        fprintf('Relative accuracy: %f\n', rel_acc_model);
        fprintf('Az: %f\n', Az_model);
            
        % Write results to the Excel file
        xlswrite(results_file, [run a0 b0 final_iter num_test_pairs abs_acc_model rel_acc_model Az_model], ...
            feature_name, sprintf('A%d', row));
        row = row + 1;
            
        % Save the mu and sigma matrices
        mumat = [mu_prior mu_post];
        sigma_post = (sigma_post + sigma_post')/2;

        save(sprintf('%s%s_%s_lesser%d_train%d_%d_run%d_mumat.mat', rel_matfolder, input, feature_name, ...
            learn_lesser, num_train_pairs_pos, num_train_pairs_neg, run), 'mumat');
        save(sprintf('%s%s_%s_lesser%d_train%d_%d_run%d_sigma.mat', rel_matfolder, input, feature_name, ...
            learn_lesser, num_train_pairs_pos, num_train_pairs_neg, run), 'sigma_post');
     end
end
        
toc;
