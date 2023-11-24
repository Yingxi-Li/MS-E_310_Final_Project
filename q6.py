import cvxpy as cp
import numpy as np

from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids

def action_hist_dependent(n, b):
    