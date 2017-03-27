# -*- coding: utf-8 -*-
"""
Logistic regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

#x1 = [3.8, 6.8, 7.9, 4.4, 1.3, 6.3, 7.3, 2.1, 3.2, 3.5]
#x2 = [3.5, 2.1, 7.7, 4.1, 7.2, 5.2, 0.5, 5.4, 9.3, 2.8]
#y = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

def count_h(x1, x2, w0, w1, w2):
    h_list = []
    for i in range(len(x1)):
        z = w0 + (w1 * x1[i]) + (w2 * x2[i])
        h = 1 / (1 + np.exp(-z))
        h_list.append(h)
    return h_list

def count_error(h_list, y):
    m = len(y)
    error = 0
    for i in range(len(y)):
        if y[i] == 1:
            error += -log(h_list[i])
        else:
            error += log(1-h_list[i])
    error = -error/m
    return error
        

def logistic_regression(x1, x2, y, alpha, iterations, error_value = 0.000001):
    w0 = 0
    w1 = 0
    w2 = 0
    
    m = len(x1)
    e = 0
    error_records = []
    h = count_h(x1, x2, w0, w1, w2)
    
    for i in range(iterations):
        
        t0 = w0 - (alpha * sum(map(lambda h, y: (h - y), h, y)) / m)
        t1 = w1 - (alpha * sum(map(lambda h, x, y: ((h - y) * x), h, x1, y)) / m)
        t2 = w2 - (alpha * sum(map(lambda h, x, y: ((h - y) * x), h, x2, y)) / m)
                     
        w0 = t0
        w1 = t1
        w2 = t2
        
        h = count_h(x1, x2, w0, w1, w2)
        error = count_error(h, y)   
        
        if i % 100 == 0:
            error_records.append(error)
        
        if abs(e - error) > error_value:
            e = error
        else:
            break
    
    error_records.append(error)
    return w0, w1, w2, error_records

def report(x1, x2, y, w0, w1, w2, error_records):
    dataset = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    positive = dataset[dataset.y == 1]
    negative = dataset[dataset.y == 0]
    
    xx = np.linspace(-5,15,100)
    xy = [(-w0 - w1*xx[i])/w2 for i in range(len(xx))]
    
    iterations = list(range(len(error_records)))
    
    print ("%s + %sx1 + %sx2" % (w0, w1, w2))
    print("error: %s" % error_records[-1])
    
    plt.figure(figsize=(6,12))
    
    plt.subplot(211)
    plt.plot(positive.x1, positive.x2, "ro")
    plt.plot(negative.x1, negative.x2, "bo")
    plt.plot(xx, xy, "g-")
    plt.title("Logistic regression")
    
    plt.subplot(212)
    plt.plot(iterations, error_records)
    plt.title("Error vs iterations")
    
    plt.savefig("logistic.png")

x1 = [np.random.normal(3, 3) for i in range(20)] + [np.random.normal(7, 3) for i in range(20)]
x2 = [np.random.normal(2, 2) for i in range(20)] + [np.random.normal(6, 3) for i in range(20)]
y = [0 for i in range(20)] + [1 for i in range(20)]

w0, w1, w2, error_records = logistic_regression(x1, x2, y, alpha=0.2, iterations=5000)
report(x1, x2, y, w0, w1, w2, error_records)


