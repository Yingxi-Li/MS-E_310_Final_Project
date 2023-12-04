import cvxpy as cp
import numpy as np
from numpy import random as rand

# Constants
m = 10 # number of goods types: 10
b = 100 * np.ones(m) # vector of initial inventory level: 1000
n = 1000 # total number of bidders: 10000
seed_val = 5

## Calculate optimal offline revenue
def get_offline_opt(A, B, Pie):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(Pie.T @ x),
                    [A @ x <= B, x <= 1, x >= 0])
    prob.solve()
    OPT_offline = prob.value
    return OPT_offline

# k: number of bids in the partial LP
# Function is tested: Gives the same dual price as solving the primal LP
# using cvxpy and asking it to provide the dual value.
# NOTE: The constraint B should be appropriately scaled by the caller beforehand
def dual_price_SLPM(A, B, Pie, k, return_y = False):
    y = cp.Variable(k)
    p = cp.Variable(m)
    prob = cp.Problem(cp.Minimize(B.T@p + cp.sum(y)),
                      [A[:,:k].T@p + y >= Pie[:k], p >= 0, y >= 0])
    prob.solve()
    if return_y:
        return p.value, y.value
    return p.value

# One-time learning algorithm
def run_OLA(A, B, Pie, k, find_dual, u_kind = None):
    if u_kind is None: # SLPM doesn't have u(s)
        p_hat = find_dual(A, k/n * B, Pie, k)
    else: # SCPM has two types of utility functions u(s)
        p_hat = find_dual(A, B, Pie, k, u_kind)
    x = np.zeros(n)
    # Below: set a tentative allocation based on dual prices, not thinking if there are enough resources
    x[k:] = np.where(A[:,k:].T@p_hat < Pie[k:], 1, 0)
    for t in range(k, n): # 1-offset due to array indexing
        if not np.all(A[:,t] * x[t] <= B - A[:,:t]@x[:t]): # not enough resources left
            x[t] = 0
    profit = Pie.T@x
    return profit, p_hat

# Create randomized bids
def make_bids(seed = seed_val):
    rand.seed(seed)
    # Fix ground truth price vector
    p_bar = rand.randint(low = 1, high = 11, size = m) / 5
    a = rand.randint(2, size = (m, n)) # Fix ground truth price vector
    pi = p_bar.T @ a + rand.normal(0, 0.2, size = n) # bid price vector
    return p_bar, a, pi

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