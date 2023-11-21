import cvxpy as cp
import numpy as np
from math import floor, sqrt
from numpy import random as rand
from numpy.linalg import norm
from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids
from q2 import run_dynamic_learn
from q4 import dual_price_SCPM

if __name__ == '__main__':
    ## Generate random bids
    p_bar, a, pi = make_bids()
    offline_OPT = get_offline_opt(a, b, pi)
    dynamic_SLPM_OPT, _ = run_dynamic_learn(a, b, pi, dual_price_SLPM)
    ratio_dynamic_SLPM = round((dynamic_SLPM_OPT / offline_OPT) * 100, 2)

    # Tune u1 parameters
    Ws = [0.1, 1, 5, 10, 50]
    revs_u1 = np.zeros(len(Ws))
    for i, w in enumerate(Ws):
        params = [w, 1]
        rev, _ = run_dynamic_learn(a, b, pi, dual_price_SCPM, 1, params)
        revs_u1[i] = round(rev, 1)

    # Tune u2
    Ws_u2, As = [1, 10], [0.1, 1, 10]
    revs_u2 = np.zeros((len(Ws_u2), len(As)))
    for i, w in enumerate(Ws_u2):
        for j, a_param in enumerate(As): # name differentiated from the matrix
            params = [w, a_param]
            rev, _ = run_dynamic_learn(a, b, pi, dual_price_SCPM, 2, params)
            revs_u2[i, j] = round(rev, 1)

    print("Optimal offline revenue: ", round(offline_OPT,1))
    print("Optimal dynamic SLPM revenue: ", round(dynamic_SLPM_OPT, 1))
    print(f"\nOptimal dynamic learning revenue, SCPM + u1, with w = {Ws}:\n", revs_u1)
    print(f"\nOptimal dynamic learning revenue, SCPM + u2, with w = {Ws_u2}, a = {As},\n" 
          "w -> row index, a -> column index:\n", revs_u2)