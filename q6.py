import cvxpy as cp
import numpy as np

from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids

def run_action_hist_dependent(A, B, Pie, find_dual, u_kind = None, params = None):
    
    OPT = get_offline_opt(A, B, Pie)
   
    dual_prices = [] # a history of dual prices, to be recorded at each update
    x = np.zeros(n)
    
    B_0 = [n * [i] for i in range(m)]
    B_j = B_0
    p_hat = 0
    
    for j in range(n):
        k = j + 1
        
        p_hat = find_dual(A, (k/(n - k)) * B, Pie, k)
        dual_prices.append(p_hat)
        
        x[j] = int(A[:,j:j+1].T @ p_hat < Pie[j:j+1])
        
        B_j = B_j - A[:,j:j+1].T * x[j]
        
    profit = Pie.T @ x
    return profit
        
if __name__ == '__main__':
    ## Generate random bids
    p_bar, a, pi = make_bids()

    # Compare offline vs online algorithms
    offline_OPT = get_offline_opt(a, b, pi)
    profits_by_k = [run_OLA(a, b, pi, k, dual_price_SLPM)[0] for k in [50, 100, 200]]

    print(f"## Q1 output. numpy random seed value={seed_val} ##")
    print(f"\nThe offline optimal revenue is ${offline_OPT}.")
    print(f"\nThe ratios of the online revenue over the offline revenue when k"
          f"\nare equal to 50, 100, 200 are respectively:"
          f"\n{[round(p / offline_OPT, 3) for p in profits_by_k]}")

    # Compare two ways of calculating the dual price
    # dual_prices_DP = get_dual_price(a,b,pi,n) # dual price calculated from dual LP

    # x = cp.Variable(n)
    # prob = cp.Problem(cp.Maximize(pi.T @ x),
    #                 [a @ x <= b, x <= 1, x >= 0])
    # prob.solve()
    # dual_prices_LP = prob.constraints[0].dual_value

    pass