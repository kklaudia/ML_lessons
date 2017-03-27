# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:58:11 2017

@author: Klaudia
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def import_data(input_file):
    data = pd.read_csv(input_file, names=["x", "y"])
    return data

def run_gradient_two(data, alpha=0.02, epoch=2):
    m = data.shape[0]
    w0 = 0
    w1 = 0
    records = pd.DataFrame(columns = ["xi", "w0", "w1"])
    
    for j in range(epoch):
        data = data.sample(frac=1).reset_index(drop=True)
        for i in range(m):
            h1 = w0 + data.x[i] * w1
            temp0 = w0 - alpha * (h1 - data.y[i])
            temp1 = w1 - alpha * (h1 - data.y[i]) * data.x[i]
            w0 = temp0
            w1 = temp1            
            records.loc[records.shape[0]] = [data.x[i], w0, w1]
    return (w0, w1, records)

def show_report_two(data, w0, w1, records):
    records = records.merge(data, left_on='xi', right_on='x', how='outer')
    records = records.rename(columns = {"y": "yi"}).drop("x", axis=1)
    records["h_xi"] = records.w0 + records.xi * records.w1
    records["point_error"] = (records.h_xi - records.yi)**2
    records["w0_change"] = (records["w0"] - records["w0"].shift(1)).abs().fillna(0)
    records["w1_change"] = (records["w1"] - records["w1"].shift(1)).abs().fillna(0)
    
    m = data.shape[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(data.x,data.y)
    final_error = (sum(((w0 + w1 * data.x) - data.y) ** 2)) / (2 * m)
    print("w0 = %s, w1 = %s" % (w0, w1))
    print("SGD result: Linear regression equation: y = %s + %sx \nError: %s" % (w0, w1, final_error))
    print("Scipy result: Linear regression equation: y = %s + %sx \nError: %s" % (intercept, slope, std_err))           

    plt.figure(figsize=(12,8))
    
    plt.subplot(231)
    plt.scatter(data.x, data.y, color="#21c66c", s=15)
    plt.plot(data.x, w0 + w1 * data.x, color="#2171c6", linewidth=2.5)
    plt.title("Linear regression", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    
    plt.subplot(232)
    iterations = records.shape[0]
    step_size = int(iterations / 5)
    indexes = list(records.index[0:iterations:step_size]) + [iterations-1]
    step_number = 1
    plt.scatter(data.x, data.y, color="#21c66c", s=15)
    for i in indexes:
        plt.plot(records.xi, records.w0[i] + records.w1[i] * records.xi, color="#2171c6", linewidth=2.5, alpha=(1/5*step_number))
        step_number +=1
    plt.title("Linear regression - step by step", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    
    plt.subplot(233)
    plt.plot(records.index, records.w0_change, color="#21c66c")
    plt.title("Changes in w0 vs time", fontsize=14)
    plt.xlabel("Time (iteration)", fontsize=12)
    plt.ylabel("W0", fontsize=12)
    
    plt.subplot(234)
    plt.plot(records.index, records.w1_change, color="#21c66c")
    plt.title("Changes in w1 vs time", fontsize=14)
    plt.xlabel("Time (iteration)", fontsize=12)
    plt.ylabel("W1", fontsize=12)
        
    plt.subplot(235)
    plt.plot(records.index, records.point_error, color="#21c66c")
    plt.title("Error value vs time", fontsize=14)
    plt.xlabel("Time (iteration)", fontsize=12)
    plt.ylabel("Error in point", fontsize=12) 

    plt.tight_layout()       
    plt.savefig("show_report_two.pdf")



# Testing

easy = pd.DataFrame({"x":[2,4,6,8,12,5,3,7,9], "y":[6,12,19,22,36,12,8,21,30]})
(easy_w0, easy_w1, easy_records) = run_gradient_two(easy, 0.005, 2)
show_report_two(easy, easy_w0, easy_w1, easy_records)
