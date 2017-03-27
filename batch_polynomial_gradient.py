# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:34:33 2017

@author: Klaudia
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_data(start, stop, points):
    x = list(np.linspace(start, stop, points))
    y = [np.sin(i) + np.random.normal(0,0.2) for i in x]
    return x,y

def count_h(x, w0, w1, w2, w3):
    h_list = []
    for i in range(len(x)):
        h = w0 + (w1 * x[i]) + (w2 * x[i]**2) + (w3 * x[i]**3)
        h_list.append(h)
    return h_list

def count_error(h_list, y):
    m = len(y)
    error = sum(map(lambda h, y: (h - y)**2, h_list, y)) / (2*m)
    return error

def polynomial_descent(x, y, alpha, iterations, error_value=0.00001):
    w0 = 1
    w1 = 1
    w2 = 1
    w3 = 1
    
    m = len(x)
    e = 0
    h = count_h(x, w0, w1, w2, w3)
    error_records = []
    
    for i in range(iterations):
        
        t0 = w0 - (alpha * sum(map(lambda h, x, y: ((h - y) * x**0), h, x, y)) / m)
        t1 = w1 - (alpha * sum(map(lambda h, x, y: ((h - y) * x**1), h, x, y)) / m)
        t2 = w2 - (alpha * sum(map(lambda h, x, y: ((h - y) * x**2), h, x, y)) / m)
        t3 = w3 - (alpha * sum(map(lambda h, x, y: ((h - y) * x**3), h, x, y)) / m)
                     
        w0 = t0
        w1 = t1
        w2 = t2
        w3 = t3
        
        h = count_h(x, w0, w1, w2, w3)
        error = count_error(h, y)   
        
        if i % 100 == 0:
            error_records.append(error)
        
        if abs(e - error) > error_value:
            e = error
        else:
            break
    
    error_records.append(error)
    return w0, w1, w2, w3, error, error_records

def report(x, y, w0, w1, w2, w3, error, error_records):
    x1 = np.arange(-np.pi*2, np.pi*2, 0.1)
    h1 = count_h(x1, w0, w1, w2, w3)
    i = range(len(error_records))
    
    print("error: %s" % error)
    
    plt.figure(figsize=(6,12))
    
    plt.subplot(211)
    plt.ylim(-2,2)
    plt.scatter(x,y)
    plt.plot(x1, h1, color="green")
    
    plt.subplot(212)
    plt.plot(i, error_records)
    
    plt.savefig("polynomial.png")


x,y = generate_data(-np.pi, np.pi, 8)
w0, w1, w2, w3, error, error_records = polynomial_descent(x, y, 0.007, 1200)
report(x, y, w0, w1, w2, w3, error, error_records)
