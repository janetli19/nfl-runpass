import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import scipy.signal as signal
from pygam import GAM, s, f

def nonparam_smooth(y,smooth_type='savgol',window=21):
    if smooth_type=='savgol':
        y[~np.isnan(y)] = signal.savgol_filter(y[~np.isnan(y)],window,2)
    return y

def poly_smooth(x,y,deg):
    nanvals = np.isnan(y)
    pfit,cov = np.polyfit(x[~nanvals],y[~nanvals],deg, cov=True)
    yp = np.poly1d(pfit)(x)
    return yp,pfit,cov

# load data
DATADIR = '    '
nfl_pbp = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')
print(nfl_pbp.shape)

# Remove overtime plays
nfl_pbp_prefiltered = nfl_pbp[nfl_pbp['game_half'] != "Overtime"]
print(nfl_pbp_prefiltered.shape)

# Code for getting posteam_won, indicator of whether team with possession won
winners = {}
uniqueGames = set(nfl_pbp_prefiltered['game_id'])
for game in uniqueGames:
    gamePlays = nfl_pbp_prefiltered[nfl_pbp_prefiltered['game_id']==game]
    finalPlay = gamePlays.iloc[(len(gamePlays)-1)]
    if finalPlay["total_home_score"] > finalPlay["total_away_score"]:
        winners[game] = finalPlay["home_team"]
    elif finalPlay["total_home_score"] < finalPlay["total_away_score"]:
        winners[game] = finalPlay["away_team"]
    else:
        winners[game] = "TIE"

posteam_won = []
margins = []
for i in range(len(nfl_pbp_prefiltered)):
    currPlay = nfl_pbp_prefiltered.iloc[i]
    if winners[currPlay["game_id"]] == currPlay["posteam"]:
        posteam_won.append(1)
    elif winners[currPlay["game_id"]] == currPlay["defteam"]:
        posteam_won.append(0)
    else:
        posteam_won.append(-1)
    margins.append(currPlay["posteam_score"] - currPlay["defteam_score"])

nfl_pbp_prefiltered["posteam_won"] = posteam_won
nfl_pbp_prefiltered["margin"] = margins

# Remove all ties
nfl_filtered = nfl_pbp_prefiltered[nfl_pbp_prefiltered['posteam_won'] != -1]
print(nfl_filtered.shape)

# Get only first down
nfl_filtered_first = nfl_filtered[nfl_filtered['down'] == 1]
print(nfl_filtered_first.shape)

# send to CSV file
# nfl_filtered_first.to_csv(r'/Users/EthanLee/nfl-runpass/nflmodeldatafirst.csv')
