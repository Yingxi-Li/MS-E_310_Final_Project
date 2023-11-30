import sys 
sys.path.append('/Users/yingxi/Desktop/MSandE_310_Final_Project')

import numpy as np
import matplotlib.pyplot as plt

# from q1 import m, b, n, get_offline_opt, dual_price_SLPM, run_OLA, make_bids
# from q6 import run_action_hist_dependent

def one_line_ci(x, y, ci, x_name, y_name, title, fig_name, do_save = False, baseline=None, xticks=None):
    """
    Plot a line graph with confidence interval.
    """

    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(6)

    # plt.plot(x, y, 'o-')
    plt.plot(x, y)
    plt.fill_between(x, np.array(y) - np.array(ci), np.array(y) + np.array(ci), alpha=0.3)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    
    if xticks:
        plt.xticks(xticks, xticks)  # Get the current locations and labels.
    
    if baseline:
        plt.axhline(y=baseline)

    if do_save:
        plt.savefig("figures/" + fig_name)
        
        
def multiple_lines_ci(x, y, ci, labels, x_name, y_name, title, fig_name, do_save = False, baseline=None):
    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(6)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i in range(len(y)):
        # plt.plot(x, y[i], 'o-', label=labels[i], c=colors[i])
        plt.plot(x, y[i], label=labels[i], c=colors[i])
        if len(baseline) == len(y):
            plt.axhline(y=baseline[i], c=colors[i])
        plt.fill_between(x, np.array(y[i]) - np.array(ci[i]), np.array(y[i]) + np.array(ci[i]), alpha=0.3)
    
    if len(baseline) == 1:
        plt.axhline(y=baseline, c="black")
        
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
        
    plt.legend()
        
    if do_save:
        plt.savefig("figures/" + fig_name)
        

# def run_multiple_trails_with_ci(k, n_trails = 10):
#     """
#     Run one configuration with multiple trails and return the mean and confidence interval.
#     """
#     ratios = []
#     for i in range(n_trails):
#         p_bar, a, pi = make_bids(seed = i)
#         offline_OPT = get_offline_opt(a, b, pi)
#         profit, _ = run_OLA(a, b, pi, k, dual_price_SLPM)
#         ratios.append(profit/offline_OPT)
#     return np.average(ratios), 1.96 * np.std(ratios) / np.sqrt(n_trails)

def run_multiple_k_with_ci(k_list, n_trails = 10):
    """
    Run multiple configuration with multiple trails and return the mean and confidence interval.
    """
    ratios = []
    cis = []
    for k in k_list:
        ratio, ci = run_multiple_trails_with_ci(k, n_trails = n_trails)
        ratios.append(ratio)
        cis.append(ci)
    return ratios, cis
        
# if __name__ == '__main__':
#     # ## q1 with confidence interval
#     # ks = [50, 100, 200]
#     # avgs, cis = run_multiple_k_with_ci(k_list = ks, n_trails = 30)
#     # one_line_ci(ks, avgs, cis, "k", "Optimality Ratio", "Optimality Ratio vs. Size of k of SLPM ", "q1.pdf", do_save = True, baseline=None, xticks=ks)
    
#     ## q6 convergence 
#     p_bar, a, pi = make_bids()
#     profit, opt_gap, offline_OPT = run_action_hist_dependent(a, b, pi, dual_price_SLPM)
#     print(profit, opt_gap[:10], offline_OPT)
#     one_line_ci(np.arange(n), opt_gap, np.zeros(len(opt_gap)), 
#                 "Iteration", "Optimality Gap", "Q6", "q6.pdf", 
#                 do_save = True, baseline=0, xticks=None)
    
