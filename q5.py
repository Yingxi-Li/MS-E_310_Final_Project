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
    dynamic_OPT_SLPM, _ = run_dynamic_learn(a, b, pi, dual_price_SLPM)
    ratio_dynamic_SLPM = round((dynamic_OPT_SLPM / offline_OPT) * 100, 2)
    
    dynamic_OPT_SCPM_u1, _ = run_dynamic_learn(a, b, pi, dual_price_SCPM, 1)
    ratio_dynamic_SCPM_u1 = round((dynamic_OPT_SCPM_u1 / offline_OPT) * 100, 2)

    dynamic_OPT_SCPM_u2, _ = run_dynamic_learn(a, b, pi, dual_price_SCPM, 2)
    ratio_dynamic_SCPM_u2 = round((dynamic_OPT_SCPM_u2 / offline_OPT) * 100, 2)

    print("Competitive ratios of three dynamic learning algorithms (in percent of offline revenue):")
    print("Dynamic SLPM: ", ratio_dynamic_SLPM)
    print("Dynamic SCPM, u(s) is sum of log's: ", ratio_dynamic_SCPM_u1)
    print("Dynamic SCPM, u(s) is sum of negative exponetials: ", ratio_dynamic_SCPM_u2)