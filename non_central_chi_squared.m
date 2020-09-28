% Author: 
% 
%     Oliver Sheridan-Methven September 2020.
% 
% Description: 
% 
%     Comparison of the non-central chi-squared to the Gaussian.

n = 1000;
nus =[1,5,10,50];
lambdas = [1,5,10,50,100,200];
fprintf('\t%snu\n', repmat(' ', 1, 5*(length(nus)-1)))
fprintf('\t%s\n', repmat('-', 1, 10*(length(nus)-1)));
fprintf('lambda\t|')
for nu=nus
fprintf('%d \t', nu);
end
fprintf('\n')
fprintf('%s\n', repmat('-', 1, 10*length(nus)));
for l=lambdas
    fprintf('%d\t| ', l);
    for nu=nus
        u = rand(1,n);
        tic;
        ncx2inv(u, nu, l);
        elapsed_ncx2 = toc/n;
        tic;
        norminv(u);
        elapsed_norm = toc/n;
        fprintf('%d \t', uint64(elapsed_ncx2/elapsed_norm));
    end
    fprintf('\n');
end
