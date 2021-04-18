import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

# load data
DATADIR = ''
full_data = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

# Code for getting posteam_won, indicator of whether team with possession won

winners = {}
uniqueGames = set(nfl_pbp['game_id'])
for game in uniqueGames:
    gamePlays = nfl_pbp[nfl_pbp['game_id']==game]
    finalPlay = gamePlays.iloc[(len(gamePlays)-1)]
    if finalPlay["total_home_score"] > finalPlay["total_away_score"]:
        winners[game] = finalPlay["home_team"]
    elif finalPlay["total_home_score"] < finalPlay["total_away_score"]:
        winners[game] = finalPlay["away_team"]
    else:
        winners[game] = "TIE"

posteam_won = []
margins = []
for i in range(len(nfl_pbp)):
    currPlay = nfl_pbp.iloc[i]
    if winners[currPlay["game_id"]] == currPlay["posteam"]:
        posteam_won.append(1)
    elif winners[currPlay["game_id"]] == currPlay["defteam"]:
        posteam_won.append(0)
    else:
        posteam_won.append(-1)
    margins.append(currPlay["posteam_score_post"] - currPlay["defteam_score_post"])

nfl_pbp["posteam_won"] = posteam_won
nfl_pbp["margin"] = margins

# to-do: fit model to find nfl_pbp["posteam_won"] using game_seconds_remaining, expected points added, margin

noTies = nfl_pbp[nfl_pbp["posteam_won"] != -1]
# pm1 = sm.GLM(noTies.posteam_won, noTies.margin, family=sm.families.Binomial())
pm1 = smf.glm('posteam_won ~ margin + game_seconds_remaining', family=sm.families.Binomial(), data=noTies)
pfit = pm1.fit()
print(pfit.summary())

# PFR Win Probability Model (using score margin, expected points added, and time remaining)

def pfrWinProb(margin, EPA, secs_left):
    return 1 - norm.cdf(0, loc = margin + EPA, scale = math.sqrt(13.45*secs_left/3600))
