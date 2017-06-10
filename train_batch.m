% Uses Bayesian logistic regression to fit the given data and outputs the
% Gaussian parameters.
% Inputs:
% traindata -- Matrix containing the input features, in which each row is
% an instance. A column of 1s has already been added.
% trainlabels -- Vector containing the class labels, 1 or -1.
% mu_prior -- The prior means of the weights.
% sigma_prior -- The prior covariance matrix of the weights.
% max_iter -- The max # of iterations for the variational method.

function [mu_post, sigma_post, sigma_post_inv] = train_batch(traindata, ...
    trainlabels, mu_prior, sigma_prior, max_iter, criterion)

    working_mu = mu_prior;
    working_sigma_inv = inv(sigma_prior);

    % learn with summary data
    [working_mu, working_sigma_inv, junk] = variational_updates_t_sum(traindata', ...
            trainlabels, working_mu, working_sigma_inv, max_iter, criterion);
        
    mu_post = working_mu;
    sigma_post = inv(working_sigma_inv);
    sigma_post_inv = working_sigma_inv;