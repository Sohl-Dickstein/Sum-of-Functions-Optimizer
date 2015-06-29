% Implements the Sum of Functions Optimizer (SFO), as described in the paper:
%     Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli
%     An adaptive low dimensional quasi-Newton sum of functions optimizer
%     arXiv preprint arXiv:1311.2115 (2013)
%     http://arxiv.org/abs/1311.2115
%
% Sample code is provided in sfo_demo.m
%
% Useful functions in this class are:
%   obj = sfo(f_df, theta, subfunction_references, varargin)
%     Initializes the optimizer class.
%     Parameters:
%       f_df - Returns the function value and gradient for a single subfunction 
%            call.  Should have the form
%                [f, dfdtheta] = f_df(theta, subfunction_references{idx},
%                                      varargin{:})
%            where idx is the index of a single subfunction.
%       theta - The initial parameters to be used for optimization.  theta can
%            be either a vector, a matrix, or a cell array with a vector or
%            matrix in every cell.  The gradient returned by f_df should have the
%            same form as theta.
%       subfunction_references - A cell array containing an identifying element
%            for each subfunction.  The elements in this list could be, eg,
%            matrices containing minibatches, or indices identifying the
%            subfunction, or filenames from which target data should be read.
%            If each subfunction corresponds to a minibatch, then the number of
%            subfunctions should be approximately
%            [number subfunctions] = sqrt([dataset size])/10.
%       varargin - Any additional parameters will be passed through to f_df
%            each time it is called.
%     Returns:
%       obj - sfo class instance.
%   theta = optimize(num_passes)
%     Optimize the objective function.
%     Parameters:
%       num_passes - The number of effective passes through
%            subfunction_references to perform.
%     Returns:
%       theta - The estimated parameter vector after num_passes of optimization.
%   check_grad()
%     Numerically checks the gradient of f_df for each element in
%     subfunction_references.
%
% Author: Jascha Sohl-Dickstein (2014)
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% ( http://creativecommons.org/licenses/by-nc/3.0/ )

classdef sfo < handle

    properties
        display = 2;
        f_df;
        args;
        max_history = 10;
        max_gradient_noise = 1;
        hess_max_dev = 1e8;
        hessian_init = 1e5;
        N;
        subfunction_references;
        % theta, in its original format;
        theta_original;
        % theta, flattented into a 1d array;
        theta;
        % theta from the previous learning step -- initialize to theta;
        theta_prior_step;
        % number of data dimensions;
        M;
        % the update steps will be rescaled by this;
        step_scale = 1;
        % 'very small' for various tasks, most importantly identifying when;
        % update steps or gradient changes are too small to be used for Hessian
        % updates without incurring large numerical errors.;
        eps = 1e-12;

        % The shortest step length allowed. Any update
        % steps shorter than this will be made this length. Set this so as to
        % prevent numerical errors when computing the difference in gradients
        % before and after a step.
        minimum_step_length = 1e-8;
        % The length of the longest allowed update step, 
        % relative to the average length of prior update steps. Takes effect 
        % after the first full pass through the data.
        max_step_length_ratio = 10;

        % the min & max dimenstionality for the subspace;
        K_min;
        K_max;
        % the current dimensionality of the subspace;
        K_current = 1;
        % obj.P holds the subspace;
        P;

        % store the minimum & maximum eigenvalue from each approximate;
        % Hessian;
        min_eig_sub;
        max_eig_sub;

        % store the total time spent in optimization, & the amount of time;
        % spent in the objective function;
        time_pass = 0.;
        time_func = 0.;

        % how many steps since the active set size was increased;
        iter_since_active_growth = 0;

        % which subfunctions are active;
        init_subf = 2;
        active;

        % the total path length traveled during optimization;
        total_distance = 0.;
        % number of function evaluations for each subfunction;
        eval_count;
        
        % theta projected into current working subspace;
        theta_proj;
        % holds the last position & the last gradient for all the objective functions;
        last_theta;
        last_df;
        % the history of theta changes for each subfunction;
        hist_deltatheta;
        % the history of gradient changes for each subfunction;
        hist_deltadf;
        % the history of function values for each subfunction;
        hist_f;
        % a flat history of all returned subfunction values for debugging/diagnostics;
        hist_f_flat = [];

        % the approximate Hessian for each subfunction is stored;
        % as dot(b(:.:.index), b(:.:.inedx).')
        b;

        % the full Hessian (sum over all the subfunctions);
        full_H = 0;

        % parameters that are passed through to f_df
        varargin_stored = {};
        
        % the predicted improvement in the total objective from the current update step
        f_predicted_total_improvement = 0;
    end

    methods

        function obj = sfo(f_df, theta, subfunction_references, varargin)
            % obj = sfo(f_df, theta, subfunction_references, varargin)
            %     Initializes the optimizer class.
            %     Parameters:
            %       f_df - Returns the function value and gradient for a single subfunction 
            %            call.  Should have the form
            %                [f, dfdtheta] = f_df(theta, subfunction_references{idx},
            %                                      varargin{:})
            %            where idx is the index of a single subfunction.
            %       theta - The initial parameters to be used for optimization.  theta can
            %            be either a vector, a matrix, or a cell array with a vector or
            %            matrix in every cell.  The gradient returned by f_df should have the
            %            same form as theta.
            %       subfunction_references - A cell array containing an identifying element
            %            for each subfunction.  The elements in this list could be, eg,
            %            matrices containing minibatches, or indices identifying the
            %            subfunction, or filenames from which target data should be read.
            %       varargin - Any additional parameters will be passed through to f_df
            %            each time it is called.
            %     Returns:
            %       obj - sfo class instance.

            obj.N = length(subfunction_references);
            obj.theta_original = theta;
            obj.theta = obj.theta_original_to_flat(obj.theta_original);
            obj.theta_prior_step = obj.theta;
            obj.M = length(obj.theta);
            obj.f_df = f_df;
            obj.varargin_stored = varargin;
            obj.subfunction_references = subfunction_references;

            subspace_dimensionality = 2.*obj.N+2;  % 2 to include current location;
            % subspace can't be larger than the full space;
            subspace_dimensionality = min([subspace_dimensionality, obj.M]);

            
            % the min & max dimenstionality for the subspace;
            obj.K_min = subspace_dimensionality;
            obj.K_max = ceil(obj.K_min.*1.5);
            obj.K_max = min([obj.K_max, obj.M]);
            % obj.P holds the subspace;
            obj.P = zeros(obj.M,obj.K_max);
            
            % store the minimum & maximum eigenvalue from each approximate;
            % Hessian;
            obj.min_eig_sub = zeros(obj.N,1);
            obj.max_eig_sub = zeros(obj.N,1);
            
            % which subfunctions are active;
            obj.active = false(obj.N,1);
            obj.init_subf = min(obj.N, obj.init_subf);
            inds = randperm(obj.N, obj.init_subf);
            obj.active(inds) = true;
            obj.min_eig_sub(inds) = obj.hessian_init;
            obj.max_eig_sub(inds) = obj.hessian_init;
            
            % number of function evaluations for each subfunction;
            obj.eval_count = zeros(obj.N,1);

            % set the first column of the subspace to be the initial;
            % theta;
            rr = sqrt(sum(obj.theta.^2));
            if rr > 0;
                obj.P(:,1) = obj.theta/rr;
            else
                % initial theta is 0 -- initialize randomly;
                obj.P(:,1) = randn(obj.M,1);
                obj.P(:,1) = obj.P(:,1) / sqrt(sum(obj.P(:,1).^2));
            end
                
            if obj.M == obj.K_max
                % if the subspace spans the full space, then (j)ust make;
                % P the identity matrix;
                if obj.display > 1;
                    fprintf('subspace spans full space');
                end
                obj.P = eye(obj.M);
                obj.K_current = obj.M+1;
            end
            
            
            % theta projected into current working subspace;
            obj.theta_proj = obj.P' * obj.theta;
            % holds the last position & the last gradient for all the objective functions;
            obj.last_theta = obj.theta_proj * ones(1,obj.N);
            obj.last_df = zeros(obj.K_max,obj.N);
            % the history of theta changes for each subfunction;
            obj.hist_deltatheta = zeros(obj.K_max,obj.max_history,obj.N);
            % the history of gradient changes for each subfunction;
            obj.hist_deltadf = zeros(obj.K_max,obj.max_history,obj.N);
            % the history of function values for each subfunction;
            obj.hist_f = ones(obj.N, obj.max_history).*nan;
            
            % the approximate Hessian for each subfunction is stored;
            % as dot(obj.b(:.:.index), obj.b(:.:.inedx).')
            obj.b = zeros(obj.K_max,2.*obj.max_history,obj.N);

            if obj.N < 25 && obj.display > 0
                fprintf( '\n\nIn experiments, performance suffered when the data was broken up into fewer\nthan 25 minibatches (and performance saturated after about 50 minibatches).\nSee Figure 2c.  You may want to use more than the current %d minibatches.\n\n', obj.N);
            end
        end


        function theta = optimize(obj, num_passes)
            % theta = optimize(num_passes)
            %     Optimize the objective function.
            %     Parameters:
            %       num_passes - The number of effective passes through
            %           subfunction_references to perform.
            %     Returns:
            %       theta - The estimated parameter vector after num_passes of optimization.

            num_steps = ceil(num_passes.*obj.N);
            for i = 1:num_steps
                if obj.display > 1
                    fprintf('pass %g, step %d,', sum(obj.eval_count)/obj.N, i);
                end
                obj.optimization_step();
                if obj.display > 1
                    fprintf('active %d/%d, sfo time %g s, func time %g s, f %f, <f> %f\n', sum(obj.active), size(obj.active, 1), obj.time_pass - obj.time_func, obj.time_func, obj.hist_f_flat(end), mean(obj.hist_f(obj.eval_count>0,1)));
                end
            end
            if obj.display > 0
                fprintf('active %d/%d, pass %g, sfo time %g s, func time %g s, <f> %f\n', sum(obj.active), size(obj.active, 1), sum(obj.eval_count)/obj.N, obj.time_pass - obj.time_func, obj.time_func, mean(obj.hist_f(obj.eval_count>0,1)));
            end

            theta = obj.theta_flat_to_original(obj.theta);
        end

        function check_grad(obj)
            % A diagnostic function to check the gradients for the subfunctions.  It
            % checks the subfunctions in random order, & the dimensions of each
            % subfunction in random order.  This way, a representitive set of
            % gradients can be checked quickly, even for high dimensional objectives.

            % step size to use for gradient check;
            small_diff = obj.eps.*1e6;
            fprintf('Testing step size %g\n', small_diff);

            for i = randperm(obj.N)
                [fl, ~, dfl] = obj.f_df_wrapper(obj.theta, i);
                ep = zeros(obj.M,1);
                dfl_obs = zeros(obj.M,1);
                dfl_err = zeros(obj.M,1);
                for j = randperm(obj.M)
                    ep(j) = small_diff;
                    fl2 = obj.f_df_wrapper(obj.theta + ep, i);
                    dfl_obs(j) = (fl2 - fl)/small_diff;
                    dfl_err(j) = dfl_obs(j) - dfl(j);
                    if abs(dfl_err(j)) > small_diff * 1e4
                        fprintf('large diff ');
                    else
                        fprintf('           ');
                    end
                    fprintf('  gradient subfunction %d, dimension %d, analytic %g, finite diff %g, error %g\n', i, j, dfl(j), dfl_obs(j), dfl_err(j));
                    ep(j) = 0.;
                end
                gerr = sqrt((sum(dfl - dfl_obs).^2));
                fprintf('subfunction %g, total L2 gradient error %g\n', i, gerr);
            end
        end


        function apply_subspace_transformation(obj,T_left,T_right)
            % Apply change-of-subspace transformation.  This function is called when;
            % the subspace is collapsed to project into the new lower dimensional;
            % subspace.;
            % T_left - The covariant subspace to subspace projection matrix.;
            % T_right - The contravariant subspace projection matrix.;

            % (note that currently T_left = T_right always since the subspace is;
            % orthogonal.  This will change if eg the code is adapted to also;
            % incorporate a 'natural gradient' based parameter space transformation.);

            [tt, ss] = size(T_left);

            % project history terms into new subspace;
            obj.last_df = (T_right.') * (obj.last_df);
            obj.last_theta = (T_left) * (obj.last_theta);
            obj.hist_deltadf = obj.reshape_wrapper(T_right.' * obj.reshape_wrapper(obj.hist_deltadf, [ss,-1]), [tt,-1,obj.N]);
            obj.hist_deltatheta = obj.reshape_wrapper(T_left * obj.reshape_wrapper(obj.hist_deltatheta, [ss, -1]), [tt,-1,obj.N]);
            % project stored hessian for each subfunction in to new subspace;
            obj.b = obj.reshape_wrapper(T_right.' * obj.reshape_wrapper(obj.b, [ss,-1]), [tt, 2.*obj.max_history,obj.N]);

            %% To avoid slow accumulation of numerical errors, recompute full_H;
            %% & theta_proj when the subspace is collapsed.  Should not be a;
            %% leading time cost.;
            % theta projected into current working subspace;
            obj.theta_proj = (obj.P.') * (obj.theta);
            % full approximate hessian;
            obj.full_H = real(obj.reshape_wrapper(obj.b,[ss,-1]) * obj.reshape_wrapper(obj.b,[ss,-1]).');
        end

        function reorthogonalize_subspace(obj)
            % check if the subspace has become non-orthogonal
            subspace_eigs = eig(obj.P.' * obj.P);
            % TODO(jascha) this may be a stricter cutoff than we need
            if max(subspace_eigs) <= 1 + obj.eps
                return
            end

            if obj.display > 2
                fprintf('subspace has become non-orthogonal.  Performing QR.\n');
            end
            [Porth, ~] = qr(obj.P(:,1:obj.K_current), 0);
            Pl = zeros(obj.K_max, obj.K_max);
            Pl(:,1:obj.K_current) = obj.P.' * Porth;
            % update the subspace;
            obj.P(:,1:obj.K_current) = Porth;
            % Pl is the projection matrix from old to new basis.  apply it to all the history;
            % terms;
            obj.apply_subspace_transformation(Pl.', Pl);
        end

        function collapse_subspace(obj, xl)
            % Collapse the subspace to its smallest dimensionality.;

            % xl is a new direction that may not be in the history yet, so we pass;
            % it in explicitly to make sure it's included.;

            if obj.display > 2
                fprintf('collapsing subspace\n');
            end

            % the projection matrix from old to new subspace;
            Pl = zeros(obj.K_max,obj.K_max);

            % yy will hold all the directions to pack into the subspace.;
            % initialize it with random noise, so that it still spans K_min;
            % dimensions even if not all the subfunctions are active yet;
            yy = randn(obj.K_max,obj.K_min);
            % the most recent position & gradient for all active subfunctions;
            % as well as the current position & gradient (which will not be saved in the history yet);
            yz = [obj.last_df(:,obj.active), obj.last_theta(:,obj.active), xl, (obj.P.') * (obj.theta)];
            yy(:,1:size(yz, 2)) = yz;
            [Pl(:,1:obj.K_min), ~] = qr(yy, 0);

            % update the subspace;
            obj.P = (obj.P) * (Pl);

            % Pl is the projection matrix from old to new basis.  apply it to all the history;
            % terms;
            obj.apply_subspace_transformation(Pl.', Pl);

            % update the stored subspace size;
            obj.K_current = obj.K_min;

            % re-orthogonalize the subspace if it's accumulated small errors
            obj.reorthogonalize_subspace();
        end


        function update_subspace(obj, x_in)
            % Update the low dimensional subspace by adding a new direction.;
            % x_in - The new vector to incorporate into the subspace.;

            if obj.K_current >= obj.M
                % no need to update the subspace if it spans the full space;
                return;
            end
            if sum(~isfinite(x_in)) > 0
                % bad vector!  bail.;
                return;
            end
            x_in_length = sqrt(sum(x_in.^2));
            if x_in_length < obj.eps
                % if the new vector is too short, nothing to do;
                return;
            end
            % make x unit length;
            xnew = x_in/x_in_length;

            % Find the component of x pointing out of the existing subspace.;
            % We need to do this multiple times for numerical stability.;
            for i = 1:3
                xnew = xnew - obj.P * (obj.P.' * xnew);
                ss = sqrt(sum(xnew.^2));
                if ss < obj.eps
                    % it barely points out of the existing subspace;
                    % no need to add a new direction to the subspace;
                    return;
                end
                % make it unit length;
                xnew = xnew / ss;
                % if it was already largely orthogonal then numerical;
                % stability will be good enough;
                % TODO replace this with a more principled test;
                if ss > 0.1
                    break;
                end
            end

            % add a new column to the subspace containing the new direction;
            obj.P(:,obj.K_current+1) = xnew;
            obj.K_current = obj.K_current + 1;

            if obj.K_current >= obj.K_max
                % the subspace has exceeded its maximum allowed size -- collapse it;
                % xl may not be in the history yet, so we pass it in explicitly to make;
                % sure it's used;
                xl = (obj.P.') * (x_in);
                obj.collapse_subspace(xl);
            end
        end

        function full_H_combined = get_full_H_with_diagonal(obj)
            % Get the full approximate Hessian, including the diagonal terms.;
            % (note that obj.full_H is stored without including the diagonal terms);

            full_H_combined = obj.full_H + eye(obj.K_max).*sum(obj.min_eig_sub(obj.active));
        end

        function f_pred = get_predicted_subf(obj, indx, theta_proj)
            % Get the predicted value of subfunction idx at theta_proj;
            % (where theat_proj is in the subspace);

            dtheta = theta_proj - obj.last_theta(:,indx);
            bdtheta = obj.b(:,:,indx).' * dtheta;
            Hdtheta = real(obj.b(:,:,indx) * bdtheta);
            Hdtheta = Hdtheta + dtheta.*obj.min_eig_sub(indx); % the diagonal contribution
            %df_pred = obj.last_df(:,indx) + Hdtheta;
            f_pred = obj.hist_f(indx,1) + obj.last_df(:,indx).' * dtheta + 0.5.*(dtheta.') * (Hdtheta);
        end


        function update_history(obj, indx, theta_proj, f, df_proj)
            % Update history of position differences & gradient differences;
            % for subfunction indx.;

            % there needs to be at least one earlier measurement from this;
            % subfunction to compute position & gradient differences.;
            if obj.eval_count(indx) > 1
                % differences in gradient & position;
                ddf = df_proj - obj.last_df(:,indx);
                ddt = theta_proj - obj.last_theta(:,indx);
                % length of gradient & position change vectors;
                lddt = sqrt(sum(ddt.^2));
                lddf = sqrt(sum(ddf.^2));

                corr_ddf_ddt = ddf.' * ddt / (lddt*lddf);

                if obj.display > 3 && corr_ddf_ddt < 0
                    fprintf('Warning!  Negative dgradient dtheta inner product.  Adding it anyway.');
                end
                if lddt < obj.eps
                    if obj.display > 2
                        fprintf('Largest change in theta too small (%g).  Not adding to history.', lddt);
                    end
                elseif lddf < obj.eps
                    if obj.display > 2
                        fprintf('Largest change in gradient too small (%g).  Not adding to history.', lddf);
                    end
                elseif abs(corr_ddf_ddt) < obj.eps
                    if obj.display > 2
                        fprintf('Inner product between dgradient and dtheta too small (%g). Not adding to history.', corr_ddf_ddt);
                    end
                else
                    if obj.display > 3
                        fprintf('subf ||dtheta|| %g, subf ||ddf|| %g, corr(ddf,dtheta) %g,', lddt, lddf, sum(ddt.*ddf)/(lddt.*lddf));
                    end

                    % shift the history by one timestep;
                    obj.hist_deltatheta(:,2:end,indx) = obj.hist_deltatheta(:,1:end-1,indx);
                    % store the difference in theta since the subfunction was last evaluated;
                    obj.hist_deltatheta(:,1,indx) = ddt;
                    % do the same thing for the change in gradient;
                    obj.hist_deltadf(:,2:end,indx) = obj.hist_deltadf(:,1:end-1,indx);
                    obj.hist_deltadf(:,1,indx) = ddf;
                end
            end

            obj.last_theta(:,indx) = theta_proj;
            obj.last_df(:,indx) = df_proj;
            obj.hist_f(indx,2:end) = obj.hist_f(indx,1:end-1);
            obj.hist_f(indx,1) = f;
        end

        
        function update_hessian(obj,indx)
            % Update the Hessian approximation for a single subfunction.;
            % indx - The index of the target subfunction for Hessian update.;

            gd = find(sum(obj.hist_deltatheta(:,:,indx).^2, 1)>0);
            num_gd = length(gd);
            if num_gd == 0
                % if no history, initialize with the median eigenvalue from full Hessian;
                if obj.display > 2
                    fprintf(' no history ');
                end
                obj.b(:,:,indx) = 0.;
                H = obj.get_full_H_with_diagonal();
                [U, ~] = obj.eigh_wrapper(H);
                obj.min_eig_sub(indx) = median(U)/sum(obj.active);
                obj.max_eig_sub(indx) = obj.min_eig_sub(indx);
                if obj.eval_count(indx) > 2
                    if obj.display > 2 || sum(obj.eval_count) < 5
                        fprintf('Subfunction evaluated %d times, but has no stored history.', obj.eval_count(indx));
                    end
                    if sum(obj.eval_count) < 5
                        fprintf('You probably need to initialize SFO with a smaller hessian_init value.  Scaling down the Hessian to try to recover.  You are better off correcting the hessian_init value though!');
                        obj.min_eig_sub(indx) = obj.min_eig_sub(indx) / 10.;
                    end
                end
                return;
            end

            % work in the subspace defined by this subfunction's history for this;
            [P_hist, ~] = qr([obj.hist_deltatheta(:,gd,indx),obj.hist_deltadf(:,gd,indx)], 0);
            deltatheta_P = P_hist.' * obj.hist_deltatheta(:,gd,indx);
            deltadf_P = P_hist.' * obj.hist_deltadf(:,gd,indx);

            %% get an approximation to the smallest eigenvalue.;
            %% This will be used as the diagonal initialization for BFGS.;
            % calculate Hessian using pinv & squared equation.  (j)ust to get;
            % smallest eigenvalue.;
            % df = H dx;
            % df^T df = dx^T H^T H dx = dx^T H^2 dx
            pdelthet = pinv(deltatheta_P);
            dd = (deltadf_P) * (pdelthet);
            H2 = (dd.') * (dd);
            [H2w, ~] = obj.eigh_wrapper(H2);
            H2w = sqrt(abs(H2w));

            % only the top ~ num_gd eigenvalues are expected to be well defined;
            H2w = sort(H2w, 'descend');
            H2w = H2w(1:num_gd);

            if min(H2w) == 0 || sum(~isfinite(H2w)) > 0
                % there was a failure using this history.  either deltadf was;
                % degenerate (0 case), | deltatheta was (non-finite case).;
                % Initialize using other subfunctions;
                H2w(:) = max(obj.min_eig_sub(obj.active));
                if obj.display > 3
                    fprintf('ill-conditioned history');
                end
            end

            obj.min_eig_sub(indx) = min(H2w);
            obj.max_eig_sub(indx) = max(H2w);

            if obj.min_eig_sub(indx) < obj.max_eig_sub(indx)/obj.hess_max_dev
                % constrain using allowed ratio;
                obj.min_eig_sub(indx) = obj.max_eig_sub(indx)/obj.hess_max_dev;
                if obj.display > 3
                    fprintf('constraining Hessian initialization');
                end
            end

            %% recalculate Hessian;
            % number of history terms;
            num_hist = size(deltatheta_P, 2);
            % the new hessian will be (b_p) * (b_p.') + eye().*obj.min_eig_sub(indx);
            b_p = zeros(size(P_hist, 2), num_hist*2);
            % step through the history;
            for hist_i = num_hist:-1:1
                s = deltatheta_P(:,hist_i);
                y = deltadf_P(:,hist_i);

                % for numerical stability;
                rscl = sqrt(sum(s.^2));
                s = s/rscl;
                y = y/rscl;

                % the BFGS step proper
                Hs = s.*obj.min_eig_sub(indx) + b_p * ((b_p.') * (s));
                term1 = y / sqrt(sum(y.*s));
                sHs = sum(s.*Hs);
                term2 = sqrt(complex(-1.)) .* Hs / sqrt(sHs);
                if sum(~isfinite(term1)) > 0 || sum(~isfinite(term2)) > 0
                    obj.min_eig_sub(indx) = max(H2w);
                    if obj.display > 1
                        fprintf('invalid bfgs history term.  should never get here!');
                    end
                    continue;
                end
                b_p(:,2*(hist_i-1)+2) = term1;
                b_p(:,2*(hist_i-1)+1) = term2;
            end

            H = real((b_p) * (b_p.')) + eye(size(b_p, 1))*obj.min_eig_sub(indx);
            % constrain it to be positive definite;
            [U, V] = obj.eigh_wrapper(H);
            if max(U) <= 0.
                % if there aren't any positive eigenvalues, then;
                % set them all to be the same conservative diagonal value;
                U(:) = obj.max_eig_sub(indx);
                if obj.display > 3
                    fprintf('no positive eigenvalues after BFGS');
                end
            end
            % set any too-small eigenvalues to the median positive;
            % eigenvalue;
            U_median = median(U(U>0));
            U(U<(max(abs(U))/obj.hess_max_dev)) = U_median;

            % the Hessian after it's been forced to be positive definite;
            H_posdef = bsxfun(@times, V, U') * V.';
            
            % now break it apart into matrices b & a diagonal term again;
            B_pos = H_posdef - eye(size(b_p, 1))*obj.min_eig_sub(indx);
            [U, V] = obj.eigh_wrapper(B_pos);
            b_p = bsxfun(@times, V, sqrt(obj.reshape_wrapper(U, [1,-1])));

            obj.b(:,:,indx) = 0.;
            obj.b(:,1:size(b_p,2),indx) = (P_hist) * (b_p);
        end


        function theta_flat = theta_original_to_flat(obj, theta_original)
            % Convert from the original parameter format into a 1d array
            % The original format can be an array, | a cell array full of
            % arrays

            if iscell(theta_original)
                theta_length = 0;
                for theta_array = theta_original(:)'
                    % iterate over cells
                    theta_length = theta_length + numel(theta_array{1});
                end
                theta_flat = zeros(theta_length,1);
                i_start = 1;
                for theta_array = theta_original(:)'
                    % iterate over cells
                    i_end = i_start + numel(theta_array{1})-1;
                    theta_flat(i_start:i_end) = theta_array{1}(:);
                    i_start = i_end+1;
                end
            else
                theta_flat = theta_original(:);
            end
        end
        function theta_new = theta_flat_to_original(obj, theta_flat)
            % Convert from a 1d array into the original parameter format.;

            if iscell(obj.theta_original)
                theta_new = cell(size(obj.theta_original));
                i_start = 1;
                jj = 1;
                for theta_array = obj.theta_original(:)'
                    % iterate over cells
                    i_end = i_start + numel(theta_array{1})-1;
                    theta_new{jj} = obj.reshape_wrapper(theta_flat(i_start:i_end), size(theta_array{1}));
                    i_start = i_end + 1;
                    jj = jj + 1;
                end                
            else
                theta_new = obj.reshape_wrapper(theta_flat, size(obj.theta_original));
            end
        end


        function [f, df_proj, df_full] = f_df_wrapper(obj, theta_in, idx)
            % A wrapper around the subfunction objective f_df, that handles the transformation;
            % into & out of the flattened parameterization used internally by SFO.;

            theta_local = obj.theta_flat_to_original(theta_in);
            % evaluate;
            t = tic();
            [f, df_full] = obj.f_df(theta_local, obj.subfunction_references{idx}, obj.varargin_stored{:});
            time_diff = toc(t);
            obj.time_func = obj.time_func + time_diff; % time spent in function evaluation;
            df_full = obj.theta_original_to_flat(df_full);
            % update the subspace with the new gradient direction;
            obj.update_subspace(df_full);
            % gradient projected into the current subspace;
            df_proj = ( obj.P.') * (df_full );
            % keep a record of function evaluations;
            obj.hist_f_flat = [obj.hist_f_flat, f];
            obj.eval_count(idx) = obj.eval_count(idx) + 1;
        end


        function indx = get_target_index(obj)
            % Choose which subfunction to update this iteration.

            % if an active subfunction has one evaluation, get a second
            % so we can have a Hessian estimate
            gd = find((obj.eval_count == 1) & obj.active);
            if ~isempty(gd)
                indx = gd(randperm(length(gd), 1));
                return
            end
            % If an active subfunction has less than two observations, then;
            % evaluate it.  We want to get to two evaluations per subfunction;
            % as quickly as possibly so that it's possible to estimate a Hessian;
            % for it
            gd = find((obj.eval_count < 2) & obj.active);
            if ~isempty(gd)
                indx = gd(randperm(length(gd), 1));
                return
            end

            % use the subfunction evaluated farthest;
            % either weighted by the total Hessian, or by the Hessian 
            % just for that subfunction
            if randn() < 0
                max_dist = -1;
                indx = -1;
                for i = 1:obj.N
                    dtheta = obj.theta_proj - obj.last_theta(:,i);
                    bdtheta = obj.b(:,:,i).' * dtheta;
                    dist = sum(bdtheta.^2) + sum(dtheta.^2)*obj.min_eig_sub(i);
                    if (dist > max_dist) && obj.active(i)
                        max_dist = dist;
                        indx = i;
                    end
                end
            else
                % from the current location, weighted by the Hessian;
                % difference between current theta & most recent evaluation;
                % for all subfunctions;
                dtheta = bsxfun(@plus, obj.theta_proj, -obj.last_theta);
                % full Hessian;
                full_H_combined = obj.get_full_H_with_diagonal();
                % squared distance;
                distance = sum(dtheta.*(full_H_combined * dtheta), 1);
                % sort the distances from largest to smallest;
                [~, dist_ord] = sort(-distance);
                % & keep only the indices that belong to active subfunctions;
                dist_ord = dist_ord(obj.active(dist_ord));
                % & choose the active subfunction from farthest away;
                indx = dist_ord(1);
                if max(distance(obj.active)) < obj.eps && sum(~obj.active)>0 && obj.eval_count(indx)>0
                    if obj.display > 2
                        fprintf('all active subfunctions evaluated here.  expanding active set.');
                    end
                    inactive = find(~obj.active);
                    indx = inactive(randperm(length(inactive), 1));
                    obj.active(indx) = true;
                end
            end

        end


        function [step_failure, f, df_proj] = handle_step_failure(obj, f, df_proj, indx)
            % Check whether an update step failed.  Update current position if it did.;

            % check to see whether the step should be a failure;
            step_failure = false;
            if ~isfinite(f) || sum(~isfinite(df_proj))>0
                % step is a failure if function | gradient is non-finite;
                step_failure = true;
            elseif obj.eval_count(indx) == 1
                % the step is a candidate for failure if it's a new subfunction, & it's;
                % much larger than expected;
                if max(obj.eval_count) > 1
                    if f > mean(obj.hist_f(obj.eval_count>1,1)) + 3*std(obj.hist_f(obj.eval_count>1,1))
                        step_failure = true;
                    end
                end
            elseif f > obj.hist_f(indx,1)
                % if this subfunction has increased in value, then look whether it's larger;
                % than its predicted value by enough to trigger a failure;
                % calculate the predicted value of this subfunction;
                f_pred = obj.get_predicted_subf(indx, obj.theta_proj);
                % if the subfunction exceeds its predicted value by more than the predicted average gain;
                % in the other subfunctions, then mark the step as a failure
                % (note that it's been about N steps since this has been evaluated, & that this subfunction can lay;
                % claim to about 1/N fraction of the objective change);
                predicted_improvement_others = obj.f_predicted_total_improvement - (obj.hist_f(indx,1) - f_pred);
                if f - f_pred > predicted_improvement_others
                    step_failure = true;
                end
            end

            if ~step_failure
                % decay the step_scale back towards 1;
                obj.step_scale = 1./obj.N + obj.step_scale .* (1. - 1./obj.N);
            else
                % shorten the step length;
                obj.step_scale = obj.step_scale / 2.;

                % the subspace may be updated during the function calls;
                % so store this in the full space;
                df = (obj.P) * (df_proj);

                [f_lastpos, df_lastpos_proj] = obj.f_df_wrapper(obj.theta_prior_step, indx);
                df_lastpos = (obj.P) * (df_lastpos_proj);

                %% if the function value exploded, then back it off until it's a;
                %% reasonable order of magnitude before adding anything to the history;
                f_pred = obj.get_predicted_subf(indx, obj.theta_proj);
                if isfinite(obj.hist_f(indx,1))
                    predicted_f_diff = abs(f_pred - obj.hist_f(indx,1));
                else
                    predicted_f_diff = abs(f - f_lastpos);
                end
                if ~isfinite(predicted_f_diff) || predicted_f_diff < obj.eps
                    predicted_f_diff = obj.eps;
                end

                for i_ls = 1:10
                    if f - f_lastpos < 10*predicted_f_diff
                        % the failed update is already with an order of magnitude;
                        % of the target update value -- no backoff required;
                        break;
                    end
                    if obj.display > 4
                        fprintf('ls %d f_diff %g predicted_f_diff %g ', i_ls, f - f_lastpos, predicted_f_diff);
                    end
                    % make the step length a factor of 100 shorter;
                    obj.theta = 0.99.*obj.theta_prior_step + 0.01.*obj.theta;
                    obj.theta_proj = (obj.P.') * (obj.theta);
                    % & recompute f & df at this new location;
                    [f, df_proj] = obj.f_df_wrapper(obj.theta, indx);
                    df = (obj.P) * (df_proj);
                end

                % we're done with function calls.  can move these back into the subspace.;
                df_proj = (obj.P.') * (df);
                df_lastpos_proj = (obj.P.') * (df_lastpos);

                if f < f_lastpos
                    % the original objective was better -- but add the newly evaluated point to the history;
                    % (j)ust so it's not a wasted function call;
                    theta_lastpos_proj = (obj.P.') * (obj.theta_prior_step);
                    obj.update_history(indx, theta_lastpos_proj, f_lastpos, df_lastpos_proj);
                    if obj.display > 2
                        fprintf('step failed, but last position was even worse ( f %g, std f %g), ', f_lastpos, std(obj.hist_f(obj.eval_count>0,1)));
                    end
                else
                    % add the change in theta & the change in gradient to the history for this subfunction;
                    % before failing over to the last position;
                    if isfinite(f) && sum(~isfinite(df_proj))==0
                        obj.update_history(indx, obj.theta_proj, f, df_proj);
                    end
                    if obj.display > 2
                        fprintf('step failed, proposed f %g, std f %g, ', f, std(obj.hist_f(obj.eval_count>0,1)));
                    end
                    if (obj.display > -1) && (sum(obj.eval_count>1) < 2)
                        fprintf([ '\nStep failed on the very first subfunction.  This is\n' ...
                                  'either due to an incorrect gradient, or a very large\n' ...
                                  'Hessian.  Try:\n' ...
                                  '   - Calling check_grad() (see README.md for details)\n' ...
                                  '   - Setting sfo.hessian_init to a larger value.\n']);
                    end
                    f = f_lastpos;
                    df_proj = df_lastpos_proj;
                    obj.theta = obj.theta_prior_step;
                    obj.theta_proj = (obj.P.') * (obj.theta);
                end
            end

            % don't let steps get so short that they don't provide any usable Hessian information;
            % TODO use a more principled cutoff here;
            obj.step_scale = max([obj.step_scale, 1e-5]);
        end


        function expand_active_subfunctions(obj, full_H_inv, step_failure)
            % expand the set of active subfunctions as appropriate;

            % power in the average gradient direction;
            df_avg = mean(obj.last_df(:,obj.active), 2);
            p_df_avg = sum(df_avg .* (full_H_inv * df_avg));
            % power of the standard error;
            ldfs = obj.last_df(:,obj.active) - df_avg*ones(1,sum(obj.active));
            num_active = sum(obj.active);
            p_df_sum = sum(sum(ldfs .* (full_H_inv * ldfs))) / num_active / (num_active - 1);
            % if the standard errror in the estimated gradient is the same order of magnitude as the gradient;
            % we want to increase the size of the active set;
            increase_desirable = (p_df_sum >= p_df_avg.*obj.max_gradient_noise);
            % increase the active set on step failure;
            increase_desirable = increase_desirable || step_failure;
            % increase the active set if we've done a full pass without updating it;
            increase_desirable = increase_desirable || (obj.iter_since_active_growth > num_active);
            % make sure all the subfunctions have enough evaluations for a Hessian approximation;
            % before bringing in new subfunctions;
            eligibile_for_increase = (min(obj.eval_count(obj.active)) >= 2);
            % one more iteration has passed since the active set was last expanded;
            obj.iter_since_active_growth = obj.iter_since_active_growth + 1;
            if increase_desirable && eligibile_for_increase && sum(~obj.active) > 0
                % the index of the new subfunction to activate;
                new_gd = find(~obj.active);
                new_gd = new_gd(randperm(length(new_gd), 1));
                if ~isempty(new_gd)
                    obj.iter_since_active_growth = 0;
                    obj.active(new_gd) = true;
                end
            end
        end


        function optimization_step(obj)
            % Perform a single optimization step.  This function is typically called by SFO.optimize().;

            time_pass_start = tic();

            %% choose an index to update;
            indx = obj.get_target_index();

            if obj.display > 2
                fprintf('||dtheta|| %g, ', sqrt(sum((obj.theta - obj.theta_prior_step).^2)));
                fprintf('index %d, last f %g, ', indx, obj.hist_f(indx,1));
                fprintf('step scale %g, ', obj.step_scale);
            end
            if obj.display > 8
                C = obj.P.' * obj.P;
                eC = eig(C);
                fprintf('mne %g, mxe %g, ', min(eC(eC>0)), max(eC));
            end

            % evaluate subfunction value & gradient at new position;
            [f, df_proj] = obj.f_df_wrapper(obj.theta, indx);

            % check for a failed update step, & adjust f, df, & obj.theta;
            % as appropriate if one occurs.;
            [step_failure, f, df_proj] = obj.handle_step_failure(f, df_proj, indx);

            % add the change in theta & the change in gradient to the history for this subfunction;
            obj.update_history(indx, obj.theta_proj, f, df_proj);

            % increment the total distance traveled using the last update;
            obj.total_distance = obj.total_distance + sqrt(sum((obj.theta - obj.theta_prior_step).^2));

            % the current contribution from this subfunction to the total Hessian approximation;
            H_pre_update = real(obj.b(:,:,indx) * obj.b(:,:,indx).');
            %% update this subfunction's Hessian estimate;
            obj.update_hessian(indx);
            % the new contribution from this subfunction to the total approximate hessian;
            H_new = real(obj.b(:,:,indx) * obj.b(:,:,indx).');
            % update total Hessian using this subfunction's updated contribution;
            obj.full_H = obj.full_H + H_new - H_pre_update;

            % calculate the total gradient, total Hessian, & total function value at the current location;
            full_df = 0.;
            for i = 1:obj.N
                dtheta = obj.theta_proj - obj.last_theta(:,i);
                bdtheta = obj.b(:,:,i).' * dtheta;
                Hdtheta = real(obj.b(:,:,i) * bdtheta);
                Hdtheta = Hdtheta + dtheta.*obj.min_eig_sub(i); % the diagonal contribution;
                full_df = full_df + Hdtheta + obj.last_df(:,i);
            end
            full_H_combined = obj.get_full_H_with_diagonal();
            % TODO - Use Woodbury identity instead of recalculating full inverse;
            full_H_inv = inv(full_H_combined);

            % calculate an update step;
            dtheta_proj = -(full_H_inv) * (full_df) .* obj.step_scale;

            dtheta_proj_length = sqrt(sum(dtheta_proj(:).^2));
            if dtheta_proj_length < obj.minimum_step_length
                dtheta_proj = dtheta_proj*obj.minimum_step_length/dtheta_proj_length;
                dtheta_proj_length = obj.minimum_step_length;
                if obj.display > 3
                    fprintf('forcing minimum step length');
                end
            end
            if sum(obj.eval_count) > obj.N && dtheta_proj_length > obj.eps
                % only allow a step to be up to a factor of max_step_length_ratio longer than the
                % average step length
                avg_length = obj.total_distance / sum(obj.eval_count);
                length_ratio = dtheta_proj_length / avg_length;
                ratio_scale = obj.max_step_length_ratio;
                if length_ratio > ratio_scale
                    if obj.display > 3
                        fprintf('truncating step length from %g to %g', dtheta_proj_length, ratio_scale*avg_length);
                    end
                    dtheta_proj_length = dtheta_proj_length/(length_ratio/ratio_scale);
                    dtheta_proj = dtheta_proj/(length_ratio/ratio_scale);
                end
            end

            % the update to theta, in the full dimensional space;
            dtheta = (obj.P) * (dtheta_proj);

            % backup the prior position, in case this is a failed step;
            obj.theta_prior_step = obj.theta;
            % update theta to the new location;
            obj.theta = obj.theta + dtheta;
            obj.theta_proj = obj.theta_proj + dtheta_proj;
            % the predicted improvement from this update step;
            obj.f_predicted_total_improvement = 0.5 .* dtheta_proj.' * (full_H_combined * dtheta_proj);

            %% expand the set of active subfunctions as appropriate;
            obj.expand_active_subfunctions(full_H_inv, step_failure);

            % record how much time was taken by this learning step;
            time_diff = toc(time_pass_start);
            obj.time_pass = obj.time_pass + time_diff;
        end
    end


    methods(Static)

        function A = reshape_wrapper(A, shape)
            % a wrapper for reshape which duplicates the numpy behavior, and sets
            % any -1 dimensions to the appropriate length

            total_dims = numel(A);
            total_assigned = prod(shape(shape>0));
            shape(shape==-1) = total_dims/total_assigned;
            A = reshape(A, shape);
        end


        function [U, V] = eigh_wrapper(A)
            % A wrapper which duplicates the order and format of the numpy
            % eigh routine.  (note, eigh further assumes symmetric matrix.  don't
            % think there's an equivalent MATLAB function?)

            % Note: this function enforces A to be symmetric

            [V,U] = eig(0.5 * (A + A'));
            U = diag(U);
        end

    end
end
