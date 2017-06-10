function [w, V, invV, logdetV, E_a, L,predprob,i] = bayes_logit_fit_nonzero_mean(X, y, muprior, a0, b0, max_iter, criterion)
% BAYES_LOGIT_FIT(X, y, V_prior) returns parpameters of a fitted logit
% model p(y = 1 | x, w) = 1 / (1 + exp(- w' * x)).
% The arguments are:
% X - input matrix, inputs x as row vectors
% y - output vector, containing either 1 or -1
% The function returns the posterior p(w1 | X, y) = N(w1 | w, V), and 
% additionally the inverse of V and ln|V| (just in case). The prior on
% p(w1) is determined by assigning it a hyperprior p(w1_i | a_i) = 
% N(w1_i | muprior_i, a_i^-1) on each of its elements, with 
% a_i = Gam(a_i | a0, b0), with parameters that make it
% uninformative. The returned vector E_a is the expectations of a_i. 
% L is the final variational bound, which is a lower bound on the log-model
% evidence.

% hyperprior parameters
if nargin<3
    a0 = 1e-2;
    b0 = 1e-4;
end; 

% equations from Bishop (2006) PRML Book + errata (!) + new stuff

% constants
[num_datapts, dim_x] = size(X);
%max_iter = 100;
an = a0 +  0.5;

% start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
% xi=ones(N, 1)*10;
% 
% lam_xi = lam(xi); %ones(N, 1) / 8;
% E_a = ones(dim_x, 1) * a0 / b0;
% w_fixedpart = 0.5 * sum(X .* repmat(y, 1, dim_x), 1)';
% invV = diag(E_a) + 2 * X' * (X .* repmat(lam_xi, 1, dim_x));%inv(sigmaprior); %
% V = inv(invV); % sigmaprior; %
% w = V * w_fixedpart;  %V*diag(E_a)*muprior +   V * w_fixedpart
% bn = b0 + 0.5 * (w .^ 2 + diag(V));
% L_last = - sum(log(1 + exp(- xi))) + sum(lam_xi .* xi .^ 2) ...
%     + 0.5 * (w' * invV * w - logdet(invV) - sum(xi)) ...
%     + dim_x * (gammaln(an) + an) - sum((b0 * an) ./ bn) - sum(an * log(bn));

xi=ones(num_datapts, 1)*0.1;

lam_xi = lam(xi); %ones(num_datapts, 1) / 8;
E_a = ones(dim_x, 1) * a0 / b0;
w_fixedpart = 0.5 * sum(X .* repmat(y, 1, dim_x), 1)'; 
invV = diag(E_a) + 2 * X' * (X .* repmat(lam_xi, 1, dim_x));% + inv(sigmaprior)*eye(size(sigmaprior)); %inv(sigmaprior); %
V = inv(invV); % sigmaprior; %
w =  V * (diag(E_a) * muprior + w_fixedpart);  %V*diag(E_a)*muprior +  V * w_fixedpart
bn = b0 + 0.5 * ((w-muprior) .^ 2 +diag(V));% 
L_last = - sum(log(1 + exp(- xi))) + sum(lam_xi .* xi .^ 2) ...
    + 0.5 * (w' * invV * w - logdet(invV) - sum(xi)) ...
    + dim_x * (gammaln(an) + an) - sum((b0 * an) ./ bn) - sum(an * log(bn));

    
for i = 1:max_iter;
    %fprintf('i in loop: %d\n', i);
    % update xi by EM-algorithm
    xi = sqrt(sum(X .* (X * (V + w * w')), 2));
    lam_xi = lam(xi);
    % update posterior parameters of a based on xi
    bn = b0 + 0.5 * ((w-muprior) .^ 2 +diag(V));%
    E_a = an ./ bn;
    % recompute posterior parameters of w
    invV = diag(E_a) + 2 * X' * (X .* repmat(lam_xi, 1, dim_x));
    V = inv(invV);
    logdetV = - logdet(invV);
    w = V * (diag(E_a) * muprior + w_fixedpart);  %V V*diag(E_a)*muprior +  * w_fixedpart;
    
    % variational bound
    L = - sum(log(1 + exp(- xi))) + sum(lam_xi .* xi .^ 2) ...
        + 0.5 * (w' * invV * w + logdetV - sum(xi)) ...
        + dim_x * (gammaln(an) + an) - sum(b0 * E_a) - sum(an * log(bn));
    % either stop if variational bound grows or change is < 0.001%
    % HACK ALARM: theoretically, the bound should never grow, and it doing
    % so points to numerical instabilities. As it seems, these start to
    % occur close to the optimal bound, which already points to a good
    % approximation.    
    if (L_last > L) %& i>5
        fprintf('warning: variational bound should grow, L_last > L, i=%d\n',i);
%        error('warning: variational bound should grow, L_last > L');
%         break  
    end
    
    if abs(L_last - L) < abs(criterion * L) & i>20
         break
    end
    L_last = L;  
    
    Lmat(i) = L; 
end;
fprintf('final i: %d\n', i);

if i == max_iter
    warning('Bayes:maxIter', ...
        'Bayesian logistic regression reached maximum number of iterations.');
end
% add constant terms to variational bound
L = L - dim_x * (gammaln(a0) + a0 * log(b0));
predprob = exp(L);

function out = lam(xi)
% returns 1 / (4 * xi) * tanh(xi / 2)
divby0_w = warning('query', 'MATLAB:divideByZero');
warning('off', 'MATLAB:divideByZero');
out = tanh(xi ./ 2) ./ (4 .* xi);
warning(divby0_w.state, 'MATLAB:divideByZero');
% fix values where xi = 0
out(isnan(out)) = 1/8;