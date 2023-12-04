import cvxpy as cp
import numpy as np
from math import floor, sqrt
from numpy import random as rand
from numpy.linalg import norm
from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids
from utils.grapher import one_line_ci, multiple_lines_ci

# dynamic learning algorithm
# params are (w,a) for the utility function in SCPM
def run_dynamic_learn(A, B, Pie, find_dual, u_kind = None, params = None, return_x = False):
    s0 = 50
    eps = s0 / n
    R = floor(np.log2(n / s0))
    dual_prices = [] # a history of dual prices, to be recorded at each update
    x = np.zeros(n)
    for r in range(1, R+1):
        l = s0 * (2**r)
        l_next = min(l * 2, n)
        h_l = eps * sqrt(n / l)
        if u_kind is None:
            p_hat = find_dual(A, (1 - h_l) * (l/n) * B, Pie, l) # SLPM, debiased
        else: # SCPM
            p_hat = find_dual(A, B, Pie, l, u_kind, params)
        dual_prices.append(p_hat)
        # set a tentative allocation based on dual prices, not thinking if there are 
        # enough resources
        x[l:l_next] = np.where(A[:,l:l_next].T@p_hat < Pie[l:l_next], 1, 0)
        for t in range(l, l_next): # 1-offset due to array indexing
            if not np.all(A[:,t] * x[t] <= B - A[:,:t]@x[:t]): # not enough resources left
                x[t] = 0
    profit = Pie.T@x
#     print(x[:100])
    
    if return_x:
          return profit, dual_prices, x
    return profit, dual_prices



if __name__ == '__main__':

    ## Generate random bids
    p_bar, a, pi = make_bids()

    offline_OPT = get_offline_opt(a, b, pi)
    OLA_OPT, _ = run_OLA(a, b, pi, 200, dual_price_SLPM) # optimal value from one-time learning
    dynamic_OPT, p_hat_list = run_dynamic_learn(a, b, pi)
    p_hat_all = dual_price_SLPM(a,b,pi,n) # dual price from solving the entire dual LP
    # Calculate the KPIs of one-time & dynamic learning algorithms
    OLA_ratio = round((OLA_OPT / offline_OPT) * 100, 2)
    dynamic_ratio = round((dynamic_OPT / offline_OPT) * 100, 2)
    # Calculate the deviations of the dual estimates w.r.t. p_bar
    p_bar_diff_L2 = np.array([norm(p_hat - p_bar) for p_hat in p_hat_list])
    p_bar_diff_inf = np.array([norm(p_hat - p_bar, np.inf) for p_hat in p_hat_list])

    # Calculate the deviations of the dual estimates w.r.t. p_hat_all
    p_hat_all_diff_L2 = np.array([norm(p_hat - p_hat_all) for p_hat in p_hat_list])
    p_hat_all_diff_inf = np.array([norm(p_hat - p_hat_all, np.inf) for p_hat in p_hat_list])

    print(f"\nWith k = 200, the one-time learning algorithm achieves {OLA_ratio}% of the "
          "offline revenue.")
    print(f"\nBy comparison, the dynamic learning algorithm recovers {dynamic_ratio}% of the "
          "offline revenue.")
    print(f"\nAs k consecutively doubles, the L2 norm of the difference between the dual estimate "
          f"and p_bar evolves as the following:\n{p_bar_diff_L2}"
          "\nMeanwhile, the L-infinity norm of the difference evolves as the following:"
          f"\n{p_bar_diff_inf}")
    print(f"\nIt appears that initially p_hat (the dual price estimate) seems to converge to p_bar "
          "\n(the ground truth vector), but it stops making progress after the first four updates. "
          "\nInteresting, the L2 and L-infinity norms of the difference between p_hat_all and p_bar are "
          f"\nrespectively {round(norm(p_hat_all - p_bar), 6)} and {round(norm(p_hat_all - p_bar, np.inf), 6)}, "
          "which are very close to that of the last terms of the two lists above.")

    print(f"\nThe above observation motivates us to probe the convergence of the dual estimate to p_hat_all, "
          "\nthe dual price calculated from solving the entire dual LP. As k consecutively doubles, the L2 norm "
          f"\nof the difference between the dual estimate and p_hat_all evolves as the following:\n{p_hat_all_diff_L2}"
          "\nMeanwhile, the L-infinity norm of the difference evolves as the following:"
          f"\n{p_hat_all_diff_inf}")
    print("\nThe takeaway is that the dual estimate converges to the dual price calculated from solving "
          "\nthe entire dual LP, but this limit point is different from p_bar, the ground truth vector. This is "
          "\nlikely because our initialization of the bid vector pi involves adding Gaussian noise which introduces bias.")
    
    multiple_lines_ci([200], [norm(p_hat_all - p_bar), norm(p_hat_all - p_bar, np.inf)], [np.zeros()], "k", "Optimality Ratio", "Optimality Ratio vs. Size of k of SLPM ", "q1.pdf", do_save = True, baseline=None, xticks=[200])
    
    pass