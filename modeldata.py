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
DATADIR = '/Users/EthanLee/Desktop/STAT 143/'
nfl = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

nfl_pbp = nfl[nfl['down'] >=3]

# Remove overtime plays
nfl_pbp_prefiltered = nfl_pbp[nfl_pbp['game_half'] != "Overtime"]

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

# Get only fourth down -> necessary b/c atm we have expected points calculator for fourth down only
nfl_filtered_fourth = nfl_filtered[nfl_filtered['down'] == 4]
print(nfl_filtered_fourth.shape)

### Basically ctrl+c ctrl+v the expected points calculator functions

# expected value of 1st down given yardline
plays = nfl_pbp[nfl_pbp['down'] == 1]
firstDown_pts = {}
for yds in range(1,100):
    firstDown_pts[yds] = np.mean(plays[plays.yardline_100 == yds].next_score_relative_to_posteam)

# cost of turnover given yardline
firstDown_cost = {}
for yds in range(1,100):
    firstDown_cost[yds] = -1 * firstDown_pts[100-yds]

## Expected points from touchdown

# 1 or 2 pt conversion
# P(1-pt conversion) * P(successful 1-pt conversion) + 2 * P(2-pt conversion) * P(successful 2-pt conversion)
one_pt_plays = np.sum(nfl_pbp.extra_point_attempt == 1)
two_pt_plays = np.sum(nfl_pbp.two_point_attempt == 1)

one_pt_prob = one_pt_plays / (one_pt_plays + two_pt_plays)
one_pt_success = np.mean(nfl_pbp[nfl_pbp.extra_point_attempt == 1].extra_point_result == 'good')
two_pt_success = np.mean(nfl_pbp[nfl_pbp.two_point_attempt == 1].two_point_conv_result == 'success')

extra_pts = one_pt_prob * one_pt_success + 2 * (1 - one_pt_prob) * two_pt_success

# expected points from touchdown
kickoff_dist = 75 # average starting position after kickoff
td_pts = 6 + extra_pts - firstDown_pts[kickoff_dist]

def get_firstdown(play_type, max_ytg=21):
    """
    Return dict of expected values of going for it, given yardline and yards to go
    """
    plays = nfl_pbp[nfl_pbp.play_type == play_type]

    # probability of 1st down given yards to go
    firstDown_prob = np.zeros(max_ytg-1)
    for ytg in range(1, max_ytg):
        firstDown_prob[ytg-1] = np.mean(plays.yards_gained[plays.ydstogo == ytg] >= ytg)

    # piecewise polynomial smooth
    firstDown_poly = poly_smooth(np.arange(1,max_ytg), firstDown_prob.copy(), deg=3)[0]
    print(play_type)
    print(firstDown_poly)
    firstDown_prob = dict(zip(range(1,max_ytg), firstDown_poly))

    # expected value of going for it, given yardline and yards to go
    goforit = {}
    for yardline in range(1, 100):
        yardline_value = {}
        for ytg in range(1, max_ytg):
            if yardline == ytg: # score touchdown
                yardline_value[ytg] = firstDown_prob[ytg]*td_pts + (1-firstDown_prob[ytg])*firstDown_cost[yardline]
            elif yardline > ytg:
                yardline_value[ytg] = firstDown_prob[ytg]*firstDown_pts[yardline-ytg] + (1-firstDown_prob[ytg])*firstDown_cost[yardline]
        goforit[yardline] = yardline_value
    return goforit

run_play = get_firstdown('run')
pass_play = get_firstdown('pass')


### Field goal

# probability of field goal given yardline
plays = nfl_pbp[nfl_pbp['field_goal_attempt']==1]
fg_prob = np.zeros(99)
for i in range(50): # prob of field goal at any field position >= 50 is set to 0
    rows = plays.yardline_100==i+1
    fg_prob[i] = np.mean(plays[rows].posteam_score_post > plays[rows].posteam_score)
    if fg_prob[i] != fg_prob[i]: # if nan
        fg_prob[i] = 0

# nonparametric smooth
fg_nonparam = nonparam_smooth(fg_prob.copy(), window=21)
fg_nonparam[fg_nonparam < 0] = 0
fg_nonparam[fg_nonparam > 1] = 1
print("field goal")
print(fg_nonparam)
fg_prob = dict(zip(range(1,100),fg_nonparam))

# expected value of field goal given field position
fg = {}
fg_value = 3 - firstDown_pts[kickoff_dist] # value of field goal
for yds in range(1, 100):
    if yds < 92: # max field position for field goal is 92
        fg[yds] = fg_prob[yds]*fg_value - (1-fg_prob[yds])*firstDown_pts[min(80, 100-(yds+8))]
    else:
        fg[yds] = -10 # small value that will never be chosen


### Punt

# expected value of punt given field position
plays = nfl_pbp[nfl_pbp.play_type == "punt"]
punt = {}
punt_dists = []
for yds in range(1, 100):
    punt_dist = np.mean(plays[plays.yardline_100==yds].kick_distance) - np.mean(plays[plays.yardline_100==yds].return_yards)
    punt_dists.append(punt_dist)
    if punt_dist != punt_dist: # if nan, then touchback
        punt[yds] = -1 * firstDown_pts[80]
    else:
        punt[yds] = -1 * firstDown_pts[min(round(100-yds+punt_dist), 80)]
print("punt")
print(punt_dists)

# print(fg)
# print(fg.values())
# print(punt)
# print(punt.values())

## Turn run_play into CSV to access in R
for i in run_play.keys():
    if i < 20:
        for j in range(i+1, 21):
            run_play[i][j] = 0

run_play_new = {}
for i in range(1, 100):
    run_play_new[i] = run_play[i].values()

run_df = pd.DataFrame(run_play_new)
# run_df.to_csv(r'/Users/EthanLee/nfl-runpass/rundf.csv')

## Turn pass_play into CSV to access in R
for i in pass_play.keys():
    if i < 20:
        for j in range(i+1, 21):
            pass_play[i][j] = 0

pass_play_new = {}
for i in range(1, 100):
    pass_play_new[i] = pass_play[i].values()

pass_df = pd.DataFrame(pass_play_new)
# pass_df.to_csv(r'/Users/EthanLee/nfl-runpass/passdf.csv')

## Add new row to nfl_filtered_fourth generating the new expected score difference based on play type
expected_margin = []
for index, row in nfl_filtered_fourth.iterrows():
    if row["play_type"] == "punt":
        expected_margin.append(row["margin"] + punt[row["yardline_100"]])
    elif row["play_type"] == "field_goal":
        expected_margin.append(row["margin"] + fg[row["yardline_100"]])
    elif row["play_type"] == "run":
        if row["ydstogo"] < 21:
            expected_margin.append(row["margin"] + run_play[row["yardline_100"]][row["ydstogo"]])
        else:
            expected_margin.append(10000)
    elif row["play_type"] == "pass":
        if row["ydstogo"] < 21:
            expected_margin.append(row["margin"] + pass_play[row["yardline_100"]][row["ydstogo"]])
        else:
            expected_margin.append(10000)
    else:
        expected_margin.append(10000)

nfl_filtered_fourth["expected_margin"] = expected_margin
nfl_final = nfl_filtered_fourth[nfl_filtered_fourth["expected_margin"] != 10000]

nfl_final.to_csv(r'/Users/EthanLee/nfl-runpass/nflmodeldata.csv')

## Failed attempts to make GLM :(
# pm1 = sm.GLM(noTies.posteam_won, noTies.margin, family=sm.families.Binomial())
# pm1 = smf.glm('posteam_won ~ margin + game_seconds_remaining', family=sm.families.Binomial(), data=noTies)
# pfit = pm1.fit()
# print(pfit.summary())

# Fit Generalized Additive Model: posteam_won ~ expected_margin + time_remaining
# model_data_X = nfl_final[['game_seconds_remaining', 'expected_margin']]
# model_data_Y = nfl_final[['posteam_won']]
# gam = GAM(s(0) + s(1), distribution = 'binomial', link = 'identity').fit(model_data_X, model_data_Y)
# print(gam.summary())

# PFR Win Probability Model (using score margin, expected points added, and time remaining)

# def pfrWinProb(margin, EPA, secs_left):
#     return 1 - norm.cdf(0, loc = margin + EPA, scale = math.sqrt(13.45*secs_left/3600))
