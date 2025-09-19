#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: mosesodeiaddai
"""

import pandas as pd
import numpy as np

class TLH:
    def __init__(self, startdate, taxrates, threshold):
        self.assets = {
            'VTI': 'SCHB', 'SCHB': 'VTI', #dictionary that maps primary tickers to alternate tickers
            'VTV': 'SCHV', 'SCHV': 'VTV'
        }
        self.taxrates = taxrates
        self.threshold = threshold
        self.taxlots = []
        self.washsaletracker = {}
        self.cash = 0.0
        self.tot_taxsavings = 0.0
        self.tradesrecord = []
        self.dailyportfoliorecords = []
        self.dailytlhloss = 0.0

    def tickerclass(self, ticker):
        return 'US Total Equity' if ticker in ['VTI', 'SCHB'] else 'US Large Cap Value'

    #computes current state of portfolio in terms of value of shares and classes and class distribution
    def portfolio_summary(self, current_prices): 
        total_val = 0.0
        ticker_shares = {t: 0.0 for t in self.assets.keys()}
        for _, ticker, shares, _, _ in self.taxlots:
            ticker_shares[ticker] += shares
            total_val += shares * current_prices[ticker]
        
        class_value = {'US Total Equity': 0.0, 'US Large Cap Value': 0.0}
        for ticker, shares in ticker_shares.items():
            class_name = self.tickerclass(ticker)
            class_value[class_name] += shares * current_prices[ticker]

        class_weight = {cls: val / total_val if total_val > 0 else 0.0 for cls, val in class_value.items()}
        return total_val, class_weight, ticker_shares

    def buyassets(self, date, cash, current_prices, reason):
        if cash <= 0: return

        self.cash += cash
        total_val, class_weight, _ = self.portfolio_summary(current_prices)
        target_classweight = {'US Total Equity': 0.80, 'US Large Cap Value': 0.20}
        
        if total_val == 0.0 and self.cash > 0:
            for ticker, targetw in {'VTI': 0.80, 'VTV': 0.20}.items():
                amount = self.cash * targetw
                price = current_prices[ticker]
                shares = amount / price
                self.taxlots.append((date, ticker, shares, price, amount))
                self.tradesrecord.append({'date': date, 'ticker': ticker, 'amount': amount, 'trade_reason': reason})
            self.cash = 0.0
        #buying primary ticker of underweight class
        elif self.cash > 0:
            best_class = None
            max_under = -np.inf
            for cls, targetw in target_classweight.items():
                under = targetw - class_weight[cls]
                if under > max_under:
                    max_under = under
                    best_class = cls
            
            if best_class:
                buying_ticker = 'VTI' if best_class == 'US Total Equity' else 'VTV'
                price = current_prices[buying_ticker]
                shares = self.cash / price
                cost_basis = self.cash #original price paid to acquire the security
                self.taxlots.append((date, buying_ticker, shares, price, cost_basis))
                self.tradesrecord.append({'date': date, 'ticker': buying_ticker, 'amount': cost_basis, 'trade_reason': reason})
                self.cash = 0.0

    def runtlh(self, date, current_prices): #function to run tax loss harvesting
        if not self.taxlots: return

        tlh_candidates = []
        for i, lot in enumerate(self.taxlots):
            purchase_date, ticker, shares, share_costbasis, _ = lot
            current_price = current_prices[ticker]
            loss_percent = (share_costbasis - current_price) / share_costbasis
            
            #checking for threshold requirements of tax loss harvesting
            if loss_percent >= self.threshold:
                alt_ticker = self.assets[ticker]
                washsale_enddate = self.washsaletracker.get(alt_ticker)
                
                if washsale_enddate is None or date > washsale_enddate:
                    tlh_candidates.append({
                        'lot_index': i,
                        'lot': lot,
                        'current_price': current_price,
                        'loss_amount': (share_costbasis - current_price) * shares,
                        'alternate_ticker': alt_ticker
                    })

        if not tlh_candidates: return

        tlh_candidates.sort(key=lambda x: x['loss_amount'], reverse=True) #sorts tlh in desc of losses
        best_candidate = tlh_candidates[0]
        
        lot_index = best_candidate['lot_index'] #removing best candidate from tax lot before sale
        _, old_ticker, selling_shares, _, tot_costbasis = self.taxlots.pop(lot_index)
        
        current_price = best_candidate['current_price']
        alt_ticker = best_candidate['alternate_ticker']
        
        sale_proceeds = selling_shares * current_price
        realized_loss = best_candidate['loss_amount']
        
        self.tradesrecord.append({'date': date, 'ticker': old_ticker, 'amount': -sale_proceeds, 'trade_reason': 'tlh_sale'})
        self.dailytlhloss += realized_loss
        self.washsaletracker[old_ticker] = date + pd.Timedelta(days=30) #tracking 30-day period after sale
        
        #using tlh proceeds from sale to reinvest in alternate tickers
        buying_shares = sale_proceeds / current_price
        self.taxlots.append((date, alt_ticker, buying_shares, current_price, sale_proceeds))
        self.tradesrecord.append({'date': date, 'ticker': alt_ticker, 'amount': sale_proceeds, 'trade_reason': 'tlh_buy'})


    def EOYsavings(self, date): #function to compute end of year savings which is added to portfolio
        tax_savings = self.dailytlhloss * self.taxrates['ordinary_income']
        self.tot_taxsavings += tax_savings
        self.dailytlhloss = 0.0
        
        if tax_savings > 0:
            current_prices = prices_loop.loc[date].to_dict()
            self.buyassets(date, tax_savings, current_prices, 'tlh_savings')

    def liquidate(self, date, current_prices): #function liquidates portfolio at end of investment period
        tot_proceeds = 0.0
        total_ltcg = 0.0
        
        for purchase_date, ticker, shares, _, cost_basis in self.taxlots:
            
            sale_price = current_prices[ticker]
            sale_proceeds = shares * sale_price
            gain = sale_proceeds - cost_basis
            holding_period = (date - purchase_date).days
            
            if holding_period >= 365:
                if gain > 0: total_ltcg += gain
            else:
                #discounting value of short-term capital gains to the long-term gain bucket
                if gain > 0: total_ltcg += gain * (self.taxrates['longterm_capitalgain'] / self.taxrates['ordinary_income']) 
                
            tot_proceeds += sale_proceeds
            self.tradesrecord.append({'date': date, 'ticker': ticker, 'amount': -sale_proceeds, 'trade_reason': 'liquidation_sale'})

        tax_paid = total_ltcg * self.taxrates['longterm_capitalgain']
        net_value = tot_proceeds - tax_paid
        
        return net_value, tax_paid, tot_proceeds

    def backtest(self, prices):
        initial_investment = 50000.0
        annual_contribution = 10000.0
        first_day = prices.index[0]
        
        self.cash = initial_investment
        self.buyassets(first_day, self.cash, prices.iloc[0].to_dict(), 'initial_deposit')
        
        #running daily scanning for tlh opportunities
        for date, row in prices.iterrows():
            #date = pd.to_datetime(date)
            current_prices = row.to_dict()
            self.runtlh(date, current_prices)
            
            # checking if it's the last trading day of the year
            is_eoy = (date.month == 12 and date.day == 31 and 
                      (prices.loc[prices.index > date].empty or prices.loc[prices.index > date].index[0].year > date.year))

            if is_eoy:
                if date.year in range(2013, 2018): #investment period
                    self.buyassets(date, annual_contribution, current_prices, 'annual_deposit')
                self.EOYsavings(date)

            total_val, _, tickershares = self.portfolio_summary(current_prices)
            record_base = {'date': date, 'portfolio_total_value': total_val}
            for ticker, shares in tickershares.items():
                self.dailyportfoliorecords.append({**record_base, 'ticker': ticker, 'shares': shares})
            
            #removing wash sale tracked tickers whose 30-day wash sale period have elapsed
            removing_tickers = [t for t, end_date in self.washsaletracker.items() if date >= end_date]
            for t in removing_tickers: del self.washsaletracker[t]

        last_day = prices.index[-1]
        last_prices = prices.iloc[-1].to_dict()
        netvalue, tax_paid, total_proceeds = self.liquidate(last_day, last_prices)
        
        return {'Net Liquidated Value': netvalue, 'Total Proceeds (Pre-Tax)': total_proceeds, 
                'Total Deposits': 100000.0, 'Total TLH Tax Savings Added': self.tot_taxsavings, 
                'Final Tax Paid on Capital Gains': tax_paid, 'Net Return': (netvalue / 100000.0) - 1}, self.dailyportfoliorecords,self.tradesrecord

#Execution
# Loading data
file_name = "price_data_from_2010_-_2022.xlsx"
prices = pd.read_excel(file_name, parse_dates=['Date'])
prices_loop = prices[(prices['Date'] >= '2013-01-01') & (prices['Date'] <= '2022-12-31')].set_index('Date')

taxrates = {'ordinary_income': 0.30, 'longterm_capitalgain': 0.15}
threshold = 0.02
TLH = TLH(prices_loop.index[0], taxrates, threshold)
tlhresults,pfolio,trade = TLH.backtest(prices_loop)
print(tlhresults)

#writing trades and portfolio records to CSV
# tradepath = "/Users/mosesodeiaddai/Downloads/Betterment/trade.xlsx"
# pfoliopath = "/Users/mosesodeiaddai/Downloads/Betterment/pfolio.xlsx"

# tradedf = pd.DataFrame(trade)
# pfoliodf = pd.DataFrame(pfolio)

# tradedf.to_excel(tradepath)
# pfoliodf.to_excel(pfoliopath)

