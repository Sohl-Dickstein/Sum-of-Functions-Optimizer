% Demonstrates usage of the Sum of Functions Optimizer (SFO) MATLAB
% package.  See sfo.m and
% https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer
% for additional documentation.
%
% Author: Jascha Sohl-Dickstein (2014)
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% ( http://creativecommons.org/licenses/by-nc/3.0/ )

function sfo_demo()
    % set model and training data parameters
    M = 20; % number visible units
    J = 10; % number hidden units
    D = 100000; % full data batch size
    N = floor(sqrt(D)/10.); % number minibatches
    % generate random training data
    v = randn(M,D);

    % create the cell array of subfunction specific arguments
    sub_refs = cell(N,1);
    for i = 1:N
        % extract a single minibatch of training data.
        sub_refs{i} = v(:,i:N:end);
    end

    % initialize parameters
    % Parameters can be stored as a vector, a matrix, or a cell array with a
    % vector or matrix in each cell.  Here the parameters are 
    % {[weight matrix], [hidden bias], [visible bias]}.
    theta_init = {randn(J,M), randn(J,1), randn(M,1)};
    % initialize the optimizer
    optimizer = sfo(@f_df_autoencoder, theta_init, sub_refs);
    % uncomment the following line to test the gradient of f_df
    %optimizer.check_grad();
    % run the optimizer for half a pass through the data
    theta = optimizer.optimize(0.5);
    % run the optimizer for another 20 passes through the data, continuing from 
    % the theta value where the prior call to optimize() ended
    theta = optimizer.optimize(20);
    % plot the convergence trace
    plot(optimizer.hist_f_flat);
    xlabel('Iteration');
    ylabel('Minibatch Function Value');
    title('Convergence Trace');
end

% define an objective function and gradient
function [f, dfdtheta] = f_df_autoencoder(theta, v)
    % [f, dfdtheta] = f_df_autoencoder(theta, v)
    %     Calculate L2 reconstruction error and gradient for an autoencoder
    %     with sigmoid nonlinearity.
    %     Parameters:
    %         theta - A cell array containing
    %              {[weight matrix], [hidden bias], [visible bias]}.
    %         v - A [# visible, # datapoints] matrix containing training data.
    %              v will be different for each subfunction.
    %     Returns:
    %         f - The L2 reconstruction error for data v and parameters theta.
    %         df - A cell array containing the gradient of f with each of the
    %              parameters in theta.

    W = theta{1};
    b_h = theta{2};
    b_v = theta{3};
    
    h = 1./(1 + exp(-bsxfun(@plus, W * v, b_h)));
    v_hat = bsxfun(@plus, W' * h, b_v);
    f = sum(sum((v_hat - v).^2)) / size(v, 2);
    dv_hat = 2*(v_hat - v) / size(v, 2);
    db_v = sum(dv_hat, 2);
    dW = h * dv_hat';
    dh = W * dv_hat;
    db_h = sum(dh.*h.*(1-h), 2);
    dW = dW + dh.*h.*(1-h) * v';
    % give the gradients the same order as the parameters
    dfdtheta = {dW, db_h, db_v};
end
