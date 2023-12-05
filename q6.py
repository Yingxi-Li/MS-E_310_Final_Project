import cvxpy as cp
import numpy as np

from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids
from q2 import run_dynamic_learn
from utils.grapher import one_line_ci, multiple_lines_ci

seed_val = 5

def run_action_hist_dependent(A, B, Pie, find_dual, u_kind = None, params = None):
    
    OPT = get_offline_opt(A, B, Pie)
   
    x = np.zeros(n)
    opt_gap = []
    
    B_j = B
    p_hat = np.zeros(m)
    
    for j in range(n):
        k = j + 1 # k == t
        
        x[j] = int(A[:,j].T @ p_hat < Pie[j])
        if not np.all(A[:,j] * x[j] <= B - A[:,:j] @ x[:j]): # not enough resources left
            x[j] = 0
            print(f"Not enough resources left at iteration {j}, setting x_j to 0")
        
        B_j = B_j - A[:,j] * x[j]
        
        if k < n:
            p_hat, y_hat = find_dual(A, (k/(n - k)) * B_j, Pie, k, return_y = True)
        
        opt_gap.append(Pie.T @ x - (k/n) * OPT)
        
    profit = Pie.T @ x
    return profit, opt_gap / OPT, OPT

def get_opt_gap(A, B, Pie, x):
    """
    Fetch the by iteration optimality gap. 
    """
    OPT = get_offline_opt(A, B, Pie)
    opt_gap = [(Pie[:k+1].T @ x[:k+1] - (k+1) * OPT / n) for k in range(n)]
    return opt_gap / OPT
    
        
if __name__ == '__main__':
    
    ## Generate random bids
    p_bar, a, pi = make_bids()
    
    ## Run action-history-dependent learning algorithm
    profit, opt_gap, offline_OPT = run_action_hist_dependent(a, b, pi, dual_price_SLPM)
    
    ## Run dynamic learning algorithm
    dynamic_OPT, p_hat_list, x = run_dynamic_learn(a, b, pi, dual_price_SLPM, return_x=True)
    
    print(get_offline_opt(a, b, pi), profit, dynamic_OPT)
    
    opt_gap2 = get_opt_gap(a, b, pi, x)
    
    print(f"## Q6 outputs see figures folder. numpy random seed value={seed_val} ##")
    
    ## Graph current objective value - k/n * OPT for action-history-dependent algorithm    
    multiple_lines_ci(x = np.arange(1, n, 1), 
                        y = [opt_gap[1:], opt_gap2[1:]], 
                        ci = [np.zeros(len(opt_gap2) - 1), np.zeros(len(opt_gap2) - 1)], 
                        labels = ["Action-history-dependent Learning Algorithm", "Dynamic Learning Algorithm"], 
                        x_name = "Number of Iteration", 
                        y_name = "Competitiveness Ratio with Offline Optimal Profit", 
                        title = "Q6: Comparison of Algorithms", 
                        fig_name = "q6.pdf", do_save = True, baseline=[0.0000000000001])
    
    print(f"\n Observe from the figure that action-history-dependent learning algorithm outperforms the dynamic learning, "
          "\n  algorithm in terms of the optimality gap as it gears the dual price towards the optimal dual price "
          f"\n at every step it takes. However, note that the dynamic learning algorithm is more computationally more efficient"
          "\n as n grows large, as it only needs to solve a dual LP for log(n) times whereas the action-history-dependent "
          f"\n learning algorithm needs to solve a dual LP for n times.")
    pass