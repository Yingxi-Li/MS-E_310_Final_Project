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
    profits = []
    
    for j in range(n):
        k = j + 1
        x[j] = int(A[:,j].T @ p_hat < Pie[j])
        if not np.all(A[:,j] * x[j] <= B - A[:,:j] @ x[:j]): # not enough resources left
            x[j] = 0
            
        if step_size == 'const':
            p_hat = p_hat + (x[j] * A[:,j] - D) / np.sqrt(n) 
        if step_size == 'sqrt':
            p_hat = p_hat + (x[j] * A[:,j] - D) / np.sqrt(k)
        if step_size == '1/k**a':
            p_hat = p_hat + (x[j] * A[:,j] - D) / k ** np.random.uniform(0.5, 1)
        if step_size == 'polyak':
            p_hat = p_hat + (x[j] * A[:,j] - D) * 0.75**(k)
        p_hat = np.maximum(p_hat, 0)
        
        p_hat_list.append(p_hat)
        profits.append(Pie.T @ x)
        opt_gap.append(Pie.T @ x - (k/n) * OPT)
        
        
    profit = Pie.T @ x
        
    return profit, opt_gap / OPT, OPT, p_hat_list, profits

def column_confidence_interval(arr):
    col_std_devs = np.std(arr, axis=0, ddof=1)
    margin_of_error = 1.96 * col_std_devs / np.sqrt(n)
    return margin_of_error


if __name__ == '__main__':
    
    ## Generate random bids
    # p_bar, a, pi = make_bids()
    
    ## Run action-history-dependent learning algorithm
    # profit, opt_gap, offline_OPT, p_hat_list = simple_fast(a, b, pi, step_size='sqrt')
    
    # p_true = dual_price_SLPM(a, b, pi, n)
    # p_gap = [np.linalg.norm(p_hat - p_true)/np.linalg.norm(p_true) for p_hat in p_hat_list]
    
    print(f"## Q8 outputs: ##")
    
    profit_y = []
    profit_ci = []
    
    profit_y_2 = []
    profit_ci_2 = []
    
    p_y = []
    p_ci = []
    
    for step_size in ['sqrt', 'const', '1/k**a', 'polyak']:
        opt_gaps = []
        p_gaps = []
        opt_gaps_2 = []
        for i in range(10):
            p_bar, a, pi = make_bids(seed=i)
            profit, opt_gap, offline_OPT, p_hat_list, profits = simple_fast(a, b, pi, step_size=step_size)
            
            p_true = dual_price_SLPM(a, b, pi, n)
            p_gap = [np.linalg.norm(p_hat - p_true) for p_hat in p_hat_list]
            p_gaps.append(p_gap)
            opt_gaps.append(opt_gap)
            opt_gaps_2.append(profits / offline_OPT)
        opt_gap_ci = column_confidence_interval(opt_gaps)
        p_gap_ci = column_confidence_interval(p_gaps)
        opt_gap_ci_2 = column_confidence_interval(opt_gaps_2)
        
        mean_opt_gaps = np.mean(opt_gaps, axis=0)
        mean_opt_gaps_2 = np.mean(opt_gaps_2, axis=0)
        
        profit_y.append(mean_opt_gaps)
        profit_ci.append(opt_gap_ci) 
       
        profit_y_2.append(mean_opt_gaps_2)
        profit_ci_2.append(opt_gap_ci_2)
        
        p_y.append(np.mean(p_gaps, axis=0))
        p_ci.append(p_gap_ci)
        
    for i, step_size in enumerate(['sqrt', 'const']):
        print(f"\n The simple and fast SGD-based learning algorithm with SGD approximation of dual price "
            f"\n achieves {100 + round(profit_y[i][-1]*100, 2)}% of the offline revenue when we use sqrt" 
            f"\n {step_size}.")
    
    # one_line_ci(x=np.arange(1, n+1, 1), 
    #             y=mean_opt_gaps, 
    #             ci=opt_gap_ci, 
    #             x_name="Number of Bids", y_name="Competitiveness Ratio with Offline Optimal Profit", 
    #             title="Convergence of Profit to Offline Optimal Profit with Adaptive Step Size", 
    #             fig_name="q8_profit_convergence_simple.pdf", do_save=True, baseline=[0.0000000000001])
    # one_line_ci(x=np.arange(1, n+1, 1), 
    #             y=np.mean(p_gaps, axis=0), 
    #             ci=p_gap_ci, 
    #             x_name="Number of Bids", y_name="Normalized Difference between Approximated and True Dual Prices", 
    #             title="Convergence of Approximated Dual Prices to True Dual Prices with Adaptive Step Size", 
    #             fig_name="q8_dual_price_convergence_simple1.pdf", do_save=True)
    
    multiple_lines_ci(x=np.arange(1, n+1, 1),
                      y=profit_y,
                      ci=profit_ci,
                      labels=["sqrt step-size", "constant step-size", "exponential decaying step-size", "geometric decaying step-size"],
                      x_name="Number of Bids", y_name="Competitiveness Ratio with Offline Optimal Profit", 
                      title="Comparison of Competitiveness Ratio with Different Step Sizes", 
                      fig_name="q8_profit_comp_stepsize.pdf", do_save=True, 
                      baseline=[0.0000000000001])
    
    multiple_lines_ci(x=np.arange(1, n+1, 1),
                      y=p_y,
                      ci=p_ci,
                      labels=["sqrt step-size", "constant step-size", "exponential decaying step-size", "geometric decaying step-size"],
                      x_name="Number of Bids", y_name="Normalized Difference between Approximated and True Dual Prices", 
                      title="Convergence of Approximated Dual Prices to True Dual Prices with Different Step Sizes", 
                      fig_name="q8_p_converge_stepsize.pdf", do_save=True, 
                      baseline=[0.0000000000001])
    
    multiple_lines_ci(x=np.arange(1, n+1, 1),
                      y=profit_y_2,
                      ci=profit_ci_2,
                      labels=["sqrt step-size", "constant step-size", "exponential decaying step-size", "geometric decaying step-size"],
                      x_name="Number of Bids", y_name="Optimality Gap", 
                      title="Comparison of Optimality Gaps with Different Step Sizes", 
                      fig_name="q8_profit_opt_gap_stepsize.pdf", do_save=True, 
                      baseline=[1])
    
    print(f"\n Convergence of the dual price vector to the true dual price vector see fugures.")
    
    print(f"\n Note that this algorithm, despite achieving lower profit, doesn't require the solving of"
        f"\n LPs. Thus, its runtime does not explode as the dimension of the LP increase. Observing the profit" 
        " \n convergence, we see that this algorithm tends to drain the resources faster towards the beginning, " 
        " \n causing the peak and the subsequent drop in the profit.")
    
    print(f"\n The approximated dual price vector exhibites desirable convergence to the true price. The " 
          " \n more aggressive weighted updating step-size causes the dual price to oscillate more " 
          " \n as the number of iteration increases.")
    
    pass
    

    
    
    
    