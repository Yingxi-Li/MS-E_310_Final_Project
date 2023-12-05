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
        if baseline and len(baseline) == len(y):
            plt.axhline(y=baseline[i], c=colors[i])
        plt.fill_between(x, np.array(y[i]) - np.array(ci[i]), np.array(y[i]) + np.array(ci[i]), alpha=0.3)
    
    if baseline and len(baseline) == 1:
        plt.axhline(y=baseline, c="black")
        
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
        
    plt.legend()
        
    if do_save:
        plt.savefig("figures/" + fig_name)
        

# 

    
