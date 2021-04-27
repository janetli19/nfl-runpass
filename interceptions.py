import pandas as pd
import numpy as np
import scipy.signal as signal

def nonparam_smooth(y,smooth_type='savgol',window=19):
    if smooth_type=='savgol':
        y[~np.isnan(y)] = signal.savgol_filter(y[~np.isnan(y)],window,2)
    return y

def poly_smooth(x,y,deg):
    nanvals = np.isnan(y)
    pfit,cov = np.polyfit(x[~nanvals],y[~nanvals],deg, cov=True)
    yp = np.poly1d(pfit)(x)
    return yp,pfit,cov

# load data
DATADIR = '/Users/EthanLee/Desktop/STAT 143/'
nfl = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

intercept_probs = np.zeros(99)
intercept_returns = np.zeros(99)
nfl_interceptions = nfl[nfl['interception'] == 1]
for i in range(99):
    nfl_int_ydline = nfl_interceptions[nfl_interceptions['yardline_100'] == i+1]
    nfl_ydline = nfl[nfl['yardline_100'] == i+1]
    intercept_probs[i] = nfl_int_ydline.shape[0]/nfl_ydline.shape[0]
    intercept_returns[i] = np.mean(nfl_int_ydline['return_yards'] - nfl_int_ydline['air_yards'])

intercept_prob_nonparam = nonparam_smooth(intercept_probs.copy(), window=21)
print(intercept_probs)
print(intercept_prob_nonparam)

intercept_return_nonparam = nonparam_smooth(intercept_returns.copy(), window=21)
print(intercept_returns)
print(intercept_return_nonparam)
