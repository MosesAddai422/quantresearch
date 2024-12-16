#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:43:37 2024

@author: mosesodeiaddai
"""

#This project uses the piecewise bootstrapping technique to build robust yield curves that can be used to price bonds. The yield rates 
# used in the model are obtained from the 2023 daily yield rates obtained from the Treasury

import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt

yieldata = pd.read_csv("/Users/mosesodeiaddai/Desktop/Ready/projects/daily-treasury-rates.csv") #2023 daily yield curve rates from the Treasury
# yieldata["Date"] = pd.to_datetime(yieldata["Date"], format="%Y-%m-%d")
# yieldata = yieldata.sort_values("Date", ascending=True)

setdate = "09/11/2023" #settlement date 
setdate = "02/02/2023"


row = yieldata[yieldata["Date"]==setdate]
if row.empty:
    print("empty row")
row = row.iloc[0]

#defining maturities & rates
mature = ["1 Mo", "2 Mo", "3 Mo","4 Mo","6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
rates = [row[m]/100 for m in mature ]

#mapping maturities to QuantLib
ql_mature = [
    (1, ql.Months), (2, ql.Months), (3, ql.Months),(4, ql.Months),(6, ql.Months),
    (1, ql.Years), (2, ql.Years), (3, ql.Years),
    (5, ql.Years), (7, ql.Years), (10, ql.Years),
    (20, ql.Years), (30, ql.Years)
]

#initializing Quantlib
cal = ql.UnitedStates(1)
setdatex = ql.Date(10,9,2023)
ql.Settings.instance().evaluationDate = setdatex

#helpers for yield curve
helpers = [
    ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)),
                         ql.Period(*tenor),
                         2, cal, ql.ModifiedFollowing, True, ql.Actual360())
    if tenor[1] in [ql.Months]
    else ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate)),
                           ql.Period(*tenor),
                           cal, ql.Annual, ql.Unadjusted, ql.Thirty360(ql.Thirty360.USA),
                           ql.Euribor6M())
    for rate, tenor in zip(rates, ql_mature)
]


#yield curve
curve = ql.PiecewiseLogCubicDiscount(setdatex, helpers, ql.Actual360())
ycurve = ql.YieldTermStructureHandle(curve)

#getting zero rates
dates = curve.dates()
zrates = [curve.zeroRate(d, ql.Actual360(), ql.Continuous).rate() for d in dates]


#Plotting yield curve
matdates = [d.to_date() for d in dates]
plt.figure(figsize=(10, 6))
plt.plot(matdates, zrates, marker='o', label='Zero Rates')
plt.title("Risk-Free Yield Curve from 2023 Treasury Data")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.grid()
plt.legend()
plt.show()

























