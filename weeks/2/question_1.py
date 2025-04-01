#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

housing_data = pd.read_csv("./data/train.csv")

X = housing_data[["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]].to_numpy()  
X = np.c_[np.ones(X.shape[0]), X]

y = housing_data["SalePrice"].to_numpy()

X_T = X.T
beta = np.linalg.inv(X_T @ X) @ X_T @ y
predictions = X @ beta

ss_res = np.sum((y - predictions) ** 2) 
ss_tot = np.sum((y - np.mean(y)) ** 2) 

r_squared = 1 - (ss_res / ss_tot)

print(r_squared)
