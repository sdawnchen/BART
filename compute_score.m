function [score] = compute_score(data, label, mu, sigma_inv, max_iterations, criterion)
%     % calculate the distribution of theta | (S, C^S, X^ij, C^ij) and
%     % theta | (X^ij, C^ij)
%     [pair_mu_post, pair_sigma_post_inv, pair_zeta1] = ...
%         variational_updates_t(data, label, mu_post, sigma_post_inv, criterion, max_iterations);
%     [pair_mu_prior, pair_sigma_prior_inv, pair_zeta2] = ...
%         variational_updates_t(data, label, mu_prior, sigma_prior_inv, criterion, max_iterations);
% 
%     % calculate log Q(C^ij | X^ij, S, C) and log Q(C^ij | X^ij)
%     log_pair_post = log_prediction(pair_zeta1, mu_post, sigma_post_inv, ...
%         pair_mu_post, pair_sigma_post_inv);
%     log_pair_prior = log_prediction(pair_zeta2, mu_prior, sigma_prior_inv, ...
%         pair_mu_prior, pair_sigma_prior_inv);
% 
%     % calculate the similarity score for the pair
%     %score = log_pair_post - log_pair_prior;
%     score = exp(log_pair_post);
    

    % calculate the distribution of theta | (S, C^S, X^ij, C^ij) and
    % theta | (X^ij, C^ij)
    [pair_mu, pair_sigma_inv, pair_xi] = ...
        variational_updates_t(data, label, mu, sigma_inv, max_iterations, criterion);

    % calculate log Q(C^ij | X^ij, S, C) and log Q(C^ij | X^ij)
    log_pair = log_prediction(pair_xi, mu, sigma_inv, pair_mu, pair_sigma_inv);

    % calculate the similarity score for the pair
    %score = log_pair_post - log_pair_prior;
    score = exp(log_pair);