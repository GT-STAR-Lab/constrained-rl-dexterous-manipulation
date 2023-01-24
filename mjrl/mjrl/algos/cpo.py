import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
# import mjrl.samplers.core as trajectory_sampler
# import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

# Import NPG
from mjrl.algos.npg_cg import NPG

class CPO(NPG):
    def __init__(self, env, policy, baseline,
                 cost_baseline,
                 kl_dist=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 normalized_step_size=0.01,
                 cost_lim=500,
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.cost_baseline = cost_baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.cost_lim = cost_lim
        self.backtrack_coeff = 0.9
        self.margin = 0
        self.margin_lr = 0.05
        self.learn_margin = True
        if save_logs: self.logger = DataLog()

    def train_from_paths(self, paths):

        EPS = 1e-6
        path_costs = [sum(p["costs"]) for p in paths]
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + EPS)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling,
        # but scaling can help with least squares
        cost_advantages = np.concatenate([path["cost_advantages"] for path in paths])
        # Advantage whitening
        cost_advantages = (cost_advantages - np.mean(cost_advantages)) / (np.std(cost_advantages) + EPS)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0] # pi_l_old

        surr_cost_before = self.CPI_surrogate(observations, actions, cost_advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages) # g
        t_gLL += timer.time() - ts

        vpg_cost_grad = self.flat_vpg(observations, actions, cost_advantages) # b

        c = path_costs[0] - self.cost_lim # can revert back to mean_cost
        rescale = len(paths[0])

        # Consider the right margin
        if self.learn_margin:
            self.margin += self.margin_lr * c
            self.margin = max(0, self.margin)

        c += self.margin
        c /= (rescale + EPS)

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        approx_g = hvp(npg_grad)
        q = np.dot(npg_grad, approx_g)

        if np.dot(vpg_cost_grad, vpg_cost_grad) <= 1e-8 and c < 0:
            # feasible and cost grad is zero---shortcut to pure TRPO update!
            w, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: CPO update!
            w = cg_solve(hvp, vpg_cost_grad, x_0=vpg_cost_grad.copy(),
                         cg_iters=self.FIM_invert_args['iters'])
            r = np.dot(w, approx_g)         # b^T H^{-1} g
            s = np.dot(w, hvp(w))            # b^T H^{-1} b
            A = q - r**2 / s                # should be always positive (Cauchy-Shwarz)
            B = 2 * self.kl_dist - c ** 2 / s      # does safety boundary intersect trust region? (positive = yes)

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                print('Alert! Attempting feasible recovery!')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                print('Alert! Attempting infeasible recovery!')

        if optim_case in [3,4]:
            lam = np.sqrt(q / (2 * self.kl_dist))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, r /c], [r/c, np.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A/B), LA)
            lam_b = proj(np.sqrt(q/(2 * self.kl_dist)), LB)
            f_a = lambda lam : -0.5 * (A / (lam + EPS) + B * lam) - r * c/(s + EPS)
            f_b = lambda lam : -0.5 * (q / (lam + EPS) + 2 * self.kl_dist * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c - r) / (s + EPS)
        else:
            lam = 0
            nu = np.sqrt(2 * self.kl_dist / (s + EPS))

        # normal step if optim_case > 0, but for optim_case =0,
        # perform infeasible recovery: step to purely decrease cost
        x = (1./(lam + EPS)) * (nu * w - npg_grad) if optim_case > 0 else nu * w

        # Policy update
        # --------------------------

        curr_params = self.policy.get_param_values() # old_params
        step = 1
        for k in range(100):
            step = self.backtrack_coeff ** k
            new_params = curr_params - step * x
            self.policy.set_param_values(new_params, set_new=True, set_old=False)
            kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
            surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
            surr_cost_after = self.CPI_surrogate(observations, actions, cost_advantages).data.numpy().ravel()[0]
            if kl_dist <= self.kl_dist and \
                (surr_after >= surr_before if optim_case > 1 else True) and \
                surr_cost_after - surr_cost_before <= max(-c, 0):
                print('Accepting new params at step %d of line search' % k)
                break

            if k == 99:
                print('Line search failed! Keeping old params.')
                step = 0.0

        new_params = curr_params - step * x
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        surr_cost_after = self.CPI_surrogate(observations, actions, cost_advantages).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log  information
        if self.save_logs:
            self.logger.log_kv('step', step)
            # self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('surr_cost_improvement', surr_cost_after - surr_cost_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate, average_cost = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                    self.logger.log_kv('average_cost', average_cost)
                except:
                    pass

        return base_stats