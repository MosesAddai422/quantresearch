#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:02:00 2024

@author: mosesodeiaddai
"""

#This project employs Monte Carlo simulation to price options with Black-Scholes as framework

import numpy as np
from sklearn.linear_model import LinearRegression

#describing parameters
s0 = 54 #current stock price
k = 85 #strike price
r = 0.065 #interest rate
sig = 0.35 #volatility
T = 1 #maturity period(yrs)
N = 100000 #no of simulations
dt = T/252 #time step
M = 252 #no of time steps

#generating random paths for stock price
np.random.seed(42)
Z = np.random.normal(0,1,(N,M)) #getting random variable
S = np.zeros((N,M))
S[:,0]=s0


#simulating stock price based on paths
for x in range(1,M):
    S[:,x] = S[:,x-1] * np.exp((r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z[:,x-1]) #stock price based on Black-Scholes model


pay = np.maximum(S[:,-1]-k,0)

#backtracking payoff to present value
for t in range(M-2, -1, -1): 
    ovstrike = S[:, t] > k
    
    if np.sum(ovstrike) > 0:  

        X = S[ovstrike, t].reshape(-1, 1)
        y = np.exp(-r * dt) * pay[ovstrike]  # discounted future payoff
        
        # fitting linear regression to estimate continuation value
        reg = LinearRegression().fit(X, y)
        nextval = reg.predict(X)
        
        # comparing next value to immediate payoff value
        pay[ovstrike] = np.maximum(pay[ovstrike], nextval)

op_price = np.mean(pay) * np.exp(-r * T)

print(f"The estimated option price is: {op_price:.2f}")