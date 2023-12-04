import cvxpy as cp
import numpy as np
from math import floor, sqrt

from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids
from utils.grapher import one_line_ci, multiple_lines_ci

def sgd_sqrt_step(f, df, x_0, eps = 1e-3, max_iter = 1000, step_size = 'sqrt'):
    """
    Implementation of SGD with step size sqrt(k)
    """
    iter = 0
    x = x_0
    
    while iter <= max_iter and np.linalg.norm(df(x)) > eps:
        if step_size == 'sqrt':
            x = x - df(x) / np.sqrt(iter + 1)
        elif step_size == 'const':
            x = x - df(x) * 0.05
        elif step_size == '1/k':
            x = x - df(x) * (1/(iter + 1))
        elif step_size == 'polyak':
            x = x - df(x) * 0.5**(iter + 1)
        iter += 1
    return x


def run_sgd_action_hist_dependent(A, B, Pie, u_kind = None, params = None):
    
    OPT = get_offline_opt(A, B, Pie)
   
    x = np.zeros(n)
    opt_gap = []
    
    B_j = B
    p_hat = np.zeros(m)
    p_hat_list = []
    
    for j in range(n):
        k = j + 1 
        
        x[j] = int(A[:,j].T @ p_hat < Pie[j])
        if not np.all(A[:,j] * x[j] <= B - A[:,:j] @ x[:j]): # not enough resources left
            x[j] = 0
        #     print(f"Not enough resources left at iteration {j}, setting x_j to 0")
        
        B_j = B_j - A[:,j] * x[j]
        
        # if k < n:
        #     p_hat, y_hat = find_dual(A, (k/(n - k)) * B_j, Pie, k, return_y = True)
        
        a = np.average(A[:,:j], axis=1)
        pi = np.average(Pie[:j+1])
        
        y_0 = np.zeros(m)
        # f = lambda y: (B/n).T @ y + np.maximum(Pie[j] - A[:,j].T @ y, 0)
        # df = lambda y: B/n - A[:,j] * (Pie[j] > A[:,j].T @ y)
        
        f = lambda y: (B/n).T @ y + np.maximum(pi - a.T @ y, 0)
        df = lambda y: B/n - a * (pi > a.T @ y)
        
        p_hat = sgd_sqrt_step(f, df, y_0)
        
        p_hat_list.append(p_hat)
        
        opt_gap.append(Pie.T @ x - (k/n) * OPT)
        
        if j % 100 == 0:
            print(f"Finished iteration {j}, profit = {Pie.T @ x}")
        
    profit = Pie.T @ x
    return profit, opt_gap / OPT, OPT, p_hat_list

def simple_fast(A, B, Pie, step_size = 'const'):
    OPT = get_offline_opt(A, B, Pie)
    opt_gap = []
    
    D = B/n
    x = np.zeros(n)
    
    p_hat = np.zeros(m)
    p_hat_list = []
    
    for j in range(n):
        k = j + 1
        x[j] = int(A[:,j].T @ p_hat < Pie[j])
        if not np.all(A[:,j] * x[j] <= B - A[:,:j] @ x[:j]): # not enough resources left
            x[j] = 0
            
        if step_size == 'const':
            p_hat = p_hat + np.sqrt(n) * (x[j] * A[:,j] - D) 
        if step_size == 'sqrt':
            p_hat = p_hat + (x[j] * A[:,j] - D) * np.sqrt(k)
        p_hat = np.maximum(p_hat, 0)
        
        p_hat_list.append(p_hat)
        opt_gap.append(Pie.T @ x - (k/n) * OPT)
        
    profit = Pie.T @ x
        
    return profit, opt_gap / OPT, OPT, p_hat_list

def column_confidence_interval(arr):
    col_std_devs = np.std(arr, axis=0, ddof=1)
    margin_of_error = 1.96 * col_std_devs / np.sqrt(n)
    return margin_of_error


if __name__ == '__main__':
    
    ## Generate random bids
    p_bar, a, pi = make_bids()
    
    ## Run action-history-dependent learning algorithm
    profit, opt_gap, offline_OPT, p_hat_list = simple_fast(a, b, pi, step_size='sqrt')
    
    p_true = dual_price_SLPM(a, b, pi, n)
    p_gap = [np.linalg.norm(p_hat - p_true) for p_hat in p_hat_list]
    
    print(f"## Q8 outputs: ##")
    
    opt_gaps = [opt_gap]
    p_gaps = [p_gap]
    for i in range(5):
        p_bar, a, pi = make_bids(seed=i*10)
        profit, opt_gap, offline_OPT, p_hat_list = run_sgd_action_hist_dependent(a, b, pi)
        
        p_true = dual_price_SLPM(a, b, pi, n)
        p_gap = [np.linalg.norm(p_hat - p_true) for p_hat in p_hat_list]
        p_gaps.append(p_gap)
        opt_gaps.append(opt_gap)
    opt_gap_ci = column_confidence_interval(opt_gaps)
    p_gap_ci = column_confidence_interval(p_gaps)
    
    mean_opt_gaps = np.mean(opt_gaps, axis=0)
    
    print(f"\n The SGD-based learning algorithm with SGD approximation of dual price "
          f"\n achieves {100 + round(mean_opt_gaps[-1]*100, 2)}% of the offline revenue")
    
    one_line_ci(x=np.arange(1, n+1, 1), 
                y=mean_opt_gaps, 
                ci=opt_gap_ci, 
                x_name="Iteration", y_name="Optimality Gap of Profit", 
                title="Convergence of Profit to Offline Optimal Profit with Adaptive Approx", 
                fig_name="q8_profit_convergence_simple.pdf", do_save=True)
    one_line_ci(x=np.arange(1, n+1, 1), 
                y=np.mean(p_gaps, axis=0), 
                ci=p_gap_ci, 
                x_name="Iteration", y_name="Distance between Approximated and True Dual Prices", 
                title="Convergence of Approximated Dual Prices to True Dual Prices with Adaptive Approx", 
                fig_name="q8_dual_price_convergence_simple.pdf", do_save=True)
    
    
    print(f"\n Convergence of the dual price vector to the true dual price vector see fugures.")
    
    pass
    

    
    
    
    