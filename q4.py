import cvxpy as cp
import numpy as np
from math import floor, sqrt
from numpy import random as rand
from numpy.linalg import norm
from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids

# concave utility function u(s)
def u(s: cp.Variable, kind, params):
    if params is None:
        w, a = 1, 1 # default parameters
    else:
        w, a = params[0], params[1]
    assert kind in [1,2]
    if kind == 1: # sum of log's
        return (w / m) * cp.sum(cp.log(s))
    else: # sum of (1 - exp(-a * s))
        return (w / m) * cp.sum(1 - cp.exp(-a * s))

# calculate the dual price vector of the SCPM
def dual_price_SCPM(A, B, Pie, k, u_kind, params):
    x = cp.Variable(n)
    s = cp.Variable(m)
    # Below: remember we are only solving the primal LP knowing the first k columns
    prob = cp.Problem(cp.Maximize(Pie[:k].T @ x[:k] + u(s, u_kind, params)),
                    [A[:,:k] @ x[:k] + s == (k/n) * B, x <= 1, x >= 0, s>=0])
    prob.solve(solver = cp.SCS)
    return prob.constraints[0].dual_value

if __name__ == '__main__':
    ## Generate random bids
    p_bar, a, pi = make_bids()

    # Compare offline vs online algorithms
    offline_OPT = get_offline_opt(a, b, pi)
    profits_by_k_SLPM = [run_OLA(a, b, pi, k, dual_price_SLPM)[0] for k in [50, 100, 200]]
    profits_by_k_SCPM_u1 = [run_OLA(a, b, pi, k, dual_price_SCPM, 1)[0] for k in [50, 100, 200]]
    profits_by_k_SCPM_u2 = [run_OLA(a, b, pi, k, dual_price_SCPM, 2)[0] for k in [50, 100, 200]]
    
    print(f"## Q4 output ##")
    print(f"\nThe offline optimal revenue is ${offline_OPT}.")
    print(f"\n1. The ratios of the online SLPM revenue over the offline revenue when k"
          f"\nare equal to 50, 100, 200 are respectively:"
          f"\n{[round(p / offline_OPT, 3) for p in profits_by_k_SLPM]}")
    print(f"2. SCPM model where u(s) is a sum of log's, with w = 1"
          f"\nThe ratios of the online SCPM revenue over the offline revenue when k"
          f"\nare equal to 50, 100, 200 are respectively:"
          f"\n{[round(p / offline_OPT, 3) for p in profits_by_k_SCPM_u1]}")
    print(f"3. SCPM model where u(s) is a sum of exponentials, with w = 1 and a = 1"
          f"\nThe ratios of the online SCPM revenue over the offline revenue when k"
          f"\nare equal to 50, 100, 200 are respectively:"
          f"\n{[round(p / offline_OPT, 3) for p in profits_by_k_SCPM_u2]}")

    pass