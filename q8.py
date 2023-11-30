import cvxpy as cp
import numpy as np
from math import floor, sqrt

from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids

def sgd_sqrt_step(f, df, x_0, eps = 1e-3, max_iter = 1000):
    """
    Implementation of SGD with step size sqrt(k)
    """
    iter = 0
    x = x_0
    
    while iter <= max_iter and np.linalg.norm(df(x)) > eps:
        x = x - df(x) * np.sqrt(iter + 1)
        iter += 1
    return x


def run_sgd_action_hist_dependent(A, B, Pie, u_kind = None, params = None):
    
    OPT = get_offline_opt(A, B, Pie)
   
    x = np.zeros(n)
    opt_gap = []
    
    B_j = B
    p_hat = np.zeros(m)
    
    for j in range(n):
        k = j + 1 
        
        x[j] = int(A[:,j].T @ p_hat < Pie[j])
        if not np.all(A[:,j] * x[j] <= B - A[:,:j] @ x[:j]): # not enough resources left
            x[j] = 0
        #     print(f"Not enough resources left at iteration {j}, setting x_j to 0")
        
        B_j = B_j - A[:,j] * x[j]
        
        # if k < n:
        #     p_hat, y_hat = find_dual(A, (k/(n - k)) * B_j, Pie, k, return_y = True)
        y_0 = np.zeros(m)
        f = lambda y: (B/n).T @ y + np.maximum(Pie[j] - A[:,j].T @ y, 0)
        df = lambda y: B/n - A[:,j] * (Pie[j] > A[:,j].T @ y)
        p_hat = sgd_sqrt_step(f, df, y_0)
        
        opt_gap.append(Pie.T @ x - (k/n) * OPT)
        
        if j % 100 == 0:
            print(f"Finished iteration {j}, profit = {Pie.T @ x}")
        
    profit = Pie.T @ x
    return profit, opt_gap / OPT, OPT

if __name__ == '__main__':
    
    ## Generate random bids
    p_bar, a, pi = make_bids()
    
    ## Run action-history-dependent learning algorithm
    profit, opt_gap, offline_OPT = run_sgd_action_hist_dependent(a, b, pi)
    
    print(f"## Q8 outputs: ##")
        
    print(f"\n The action-history-dependent learning algorithm with SGD approximation of dual price "
          f"\n achieves {round((profit / offline_OPT) * 100, 2)}% of the offline revenue")
    
    pass
    

    
    
    
    