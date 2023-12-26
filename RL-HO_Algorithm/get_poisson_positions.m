function [positions] =  get_poisson_positions(lambda, size_x, size_y)

% size_x=10;
% size_y=10;
% lambda in unit/m^2
lambda_tot=lambda*size_x*size_y;
N_poisson=poissrnd(lambda_tot);
positions = [rand(N_poisson,1)*size_x, rand(N_poisson,1)*size_y];
% figure
% plot(pos_x, pos_y,'.')
% axis([0 size_x 0 size_y])