# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 23:06:37 2021

@author: sushma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

superman = pd.read_csv('AAPL Historical Data.csv',usecols=[0,1,2,3,4])

POHL_avg=superman[['Price','open','High','Low']].mean(axis=1)

obs=np.array(1,len(superman)1+,1)

plt.plot(obs,PHOL_avg,'r',label="MY FIRST PLOT")