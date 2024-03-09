#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:34:02 2023

@author: Marianne2
"""
import matplotlib.pyplot as plt
from numpy import *
#function from Task 4
def f(x_1,x_2):
    return np.array([
    x_1**3 - 3*(x_1)*(x_2**2)-1,
     3*(x_1**2)*(x_2) - (x_2)**3])
def df(x1,x2): #jacobian matrix
   df= sp.zeros(len(f), len(variables))
   for i, fi in enumerate(f):
       for j, var in enumerate(variables):
           df[i, j] = sp.diff(fi, var)
   return df
    
#defining fractal2D
class fractal2D:
    def __init__(self,f, df,max_iter=100):
        self.f = f
        self.df = df
        self.max_iter = max_iter
        self.zeroes = []
    #newton's method iterated for max_iter iterations
    def newton_method(self,x0):
        x = x0
        for i in range(max_iter):
            delta_x = np.linalg.solve(df(x), -f(x))
            x = x + delta_x
            if np.linalg.norm(delta_x) < tol:
                break
        return x
        #x = x0
       # for i in range(self.max_iter):
           # x = x-(f(x)/f_prime(x))
        #return x
    def zeroes(self,x0):
        zero = newton_method(x0)
        #finding the zeroes we compare other points to, deciding which zeroes we append to the list
        #unsure about this part
        if zero < 1e-6: #close enough to value =0
            zeroes.append(x)
        #returns the index of the closest zero
        for i in range(max_iter):  
            closest_zero_index = min(range(len(zeroes)), key=lambda i: abs(x - found_zeroes[i]))
            #key=lambda i: abs(x - found_zeroes[i]): This part defines a key function that takes an index i and returns the absolute difference between x and the element at index i in found_zeroes. The abs function is used to ensure a positive value.
        return closest_zero_index
    #if no convergence, returns -1
        return -1

    def plot(x_0):
       x_values = np.linspace(a, b, N)
       y_values = np.linspace(c, d, N)
       X,Y = np.meshgrid(x_values, y_values)
       A[i,j] = newtons_method(x0)
       fig = plt.figure(figsize=(9, 6))
       plt.imshow(Z, origin="lower")
       plt.pcolor()
       plt.locator_params(axis='y', nbins=4)
       plt.locator_params(axis='x', nbins=4)
       plt.show()



























