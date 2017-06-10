% For generating learning curves, runs simulations for baseline and BART
% learning both "lesser" relations and "greater" relations.

input = 'ratings';%'topics';%'leuven';%
if strcmp(input, 'ratings') || strcmp(input, 'leuven')
    num_train_pairs_pos_vec = [10 20];%[1:10 20:20:100];
else
    num_train_pairs_pos_vec = 10:10:100;
end
train_vec_length = length(num_train_pairs_pos_vec);

for i = 1 : train_vec_length
    learn_test_relations(input, 0, 0, num_train_pairs_pos_vec(i));    % baseline with "lesser" relations
    learn_test_relations(input, 0, 1, num_train_pairs_pos_vec(i));    % baseline with "greater" relations
    learn_test_relations(input, 1, 0, num_train_pairs_pos_vec(i));    % BART with "lesser" relations
    learn_test_relations(input, 1, 1, num_train_pairs_pos_vec(i));    % BART with "greater" relations
end