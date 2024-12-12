#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:08:55 2024

@author: mosesodeiaddai
"""
#This project employs the use of Principal Component Analysis(PCA) and kmeans for optimal hedge identification

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, MiniBatchKMeans,KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

#Getting data and computing change in adjusted closing stock prices
stockdata = pd.read_csv('/Users/mosesodeiaddai/Desktop/Ready/projects/new_stock_data.csv', index_col=7)

#adding more features
stockdata['rolling_mean'] = stockdata['adjclose'].rolling(window=5).mean()
stockdata['rolling_std'] = stockdata['adjclose'].rolling(window=5).std()
stockdata['lag_1'] = stockdata['adjclose'].shift(1)

#removing NA and infinity values
returns = stockdata[['adjclose','rolling_mean','rolling_std','lag_1']].pct_change().dropna()
#returns = returns.set_index(['date', 'Stock Symbol'])
returns['rolling_std'] = returns['rolling_std'].replace([np.inf, -np.inf], np.nan)
returns = returns.dropna() 

scale = StandardScaler()
screturns = scale.fit_transform(returns)

#Employing PCA to reduce number of features while retaining data's variation
pca = PCA(n_components=0.95) #maintaining 95% of variance
pca_vals = pca.fit_transform(screturns)

# Explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("Explained Variance Ratio:", explained_variance)

# Visualizing explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

#Clustering stocks

#DBSCAN
# dbscan = DBSCAN(eps=0.1, min_samples=10)
# clusters = dbscan.fit_predict(pca_vals)

#kmeans - miniBatch
# kmeans = MiniBatchKMeans(n_clusters=6, batch_size=10000)  
# clusters = kmeans.fit_predict(pca_vals)

#kmeans - original
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(pca_vals)

#adding clusters to dataset
returns['Cluster'] = clusters


#Visualizing clusters
#2d
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_vals[:, 0], y=pca_vals[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters in Principal Component Space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()

#3d
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_vals[:, 0], pca_vals[:, 1], pca_vals[:, 2], c=clusters, cmap='viridis', s=50)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('Clusters in 3D Space')
plt.show()

#checking assets in each cluster
k=6
for cluster in range(k):
    print(f"\nAssets in Cluster {cluster}:")
    print(returns[returns['Cluster'] == cluster].index.tolist())


#Testing Hedges
target = 0  
clust_assets = returns[returns['Cluster'] == target].index.unique()
clust_data = stockdata.loc[clust_assets]
#num_data = clust_data[['volume', 'open', 'close', 'high', 'low', 'adjclose', 'rolling_mean', 'rolling_std', 'lag_1']]
num_data = clust_data[['volume', 'open', 'low', 'adjclose',  'lag_1']]

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Cluster Assets')
plt.show()


#Building hedging portfolio
def optimizer(covmatrix, targetind):
    numassets = covmatrix.shape[0]
    
    #minimizing variance
    def var(weights):
        return weights@covmatrix@weights
    
    #setting constraints
    const = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1) for _ in range(numassets)] 
    
    guess = np.ones(numassets)/numassets
    
    output = minimize(var,guess,constraints=const,bounds=bounds)
    return output.x

#computing covariance matrix
# num_data = num_data.dropna()
# num_data.replace([np.inf, -np.inf], np.nan, inplace=True)
# num_data_cleaned = num_data.dropna()

#
covmatrix = num_data.pct_change().dropna().cov()

#optimizing portfolio for a specific asset cluster
targetind = 0
optweights = optimizer(covmatrix, targetind)
print("Optimal Hedge Portfolio Weights:", optweights)











































