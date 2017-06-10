threshold = 0:0.01:1;
num_correct_model_abs = 0;
num_correct_model_rel = 0;

num_correct_model_FA = zeros(length(threshold), 1);
num_correct_model_HIT = zeros(length(threshold), 1);

for test = 1 : num_test_pairs
    % Test the positive pair
    pair_vec_pos = test_pos_vecs(test, :);
    pred_prob_pos = compute_score(pair_vec_pos', 1, mu_post, sigma_inv_post, max_iter, criterion);
    
    if pred_prob_pos > 0.5
        num_correct_model_abs = num_correct_model_abs + 1;
    end
    
    num_correct_model_HIT(pred_prob_pos > threshold) = ...
        num_correct_model_HIT(pred_prob_pos > threshold) + 1;
    
    % Test the negative pair
    pair_vec_neg = test_pos_vecs(test, [numfeats + 1 : end, 1 : numfeats]);
    pred_prob_neg = compute_score(pair_vec_neg', 1, mu_post, sigma_inv_post, max_iter, criterion);
    
    if pred_prob_neg < 0.5
        num_correct_model_abs = num_correct_model_abs + 1;
    end
    
    num_correct_model_FA(pred_prob_neg > threshold) = ...
        num_correct_model_FA(pred_prob_neg > threshold) + 1;
    
    if pred_prob_pos > pred_prob_neg
        num_correct_model_rel = num_correct_model_rel + 1;
    end
    
end

FA_model = num_correct_model_FA / num_test_pairs;
HIT_model = num_correct_model_HIT / num_test_pairs;

abs_acc_model = num_correct_model_abs / (num_test_pairs * 2);
rel_acc_model = num_correct_model_rel/num_test_pairs;
Az_model = trapz(flipud([1; FA_model; 0]), flipud([1; HIT_model; 0]));
