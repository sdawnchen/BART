function [reversed] = reverse_roles_mu(orig_mu, numfeats)
reversed = [orig_mu(numfeats + 1 : end); orig_mu(1 : numfeats)];
