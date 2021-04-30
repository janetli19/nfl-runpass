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
nfl = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

# only look at third and fourth down data
nfl_pbp = nfl[nfl['down'] >=3]

# Remove overtime plays (our model will only consider regular time remaining)
nfl_pbp_prefiltered = nfl_pbp[nfl_pbp['game_half'] != "Overtime"]

## Code for getting posteam_won, indicator of whether team with possession won
# first, get the winner of every game by id and store in 'winners'
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

# next, calculate posteam_won and the score margin for each play
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

# Remove all ties (this allows us to strictly have a binary indicator of whether or not a team won
nfl_filtered = nfl_pbp_prefiltered[nfl_pbp_prefiltered['posteam_won'] != -1]

# Store the filtered data as a CSV file to be accessed in R when training the models
# nfl_filtered.to_csv(r'/Users/EthanLee/nfl-runpass/nflmodeldata.csv')

### Some additional code to calculate the probabilities of success for different actions and other necessary numbers

## Expected points from touchdown

# 1 or 2 pt conversion
# P(1-pt conversion) * P(successful 1-pt conversion) + 2 * P(2-pt conversion) * P(successful 2-pt conversion)
one_pt_plays = np.sum(nfl.extra_point_attempt == 1)
two_pt_plays = np.sum(nfl.two_point_attempt == 1)

one_pt_prob = one_pt_plays / (one_pt_plays + two_pt_plays)
one_pt_success = np.mean(nfl[nfl.extra_point_attempt == 1].extra_point_result == 'good')
two_pt_success = np.mean(nfl[nfl.two_point_attempt == 1].two_point_conv_result == 'success')

extra_pts = one_pt_prob * one_pt_success + 2 * (1 - one_pt_prob) * two_pt_success

# This value was manually copied into the R code as the expected value of extra points added on to a touchdown
print(extra_pts)

## find probability of field goal success using our 3rd/4th down data

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

## find the expected punt distance given yardline, using our 3rd/4th down data

plays = nfl_pbp[nfl_pbp.play_type == "punt"]
punt_dists = []
for yds in range(1, 100):
    punt_dist = np.mean(plays[plays.yardline_100==yds].kick_distance) - np.mean(plays[plays.yardline_100==yds].return_yards)
    if punt_dist != punt_dist:
        punt_dists.append(0)
    else:
        punt_dists.append(punt_dist)

punt_fg_probs = {'punt': punt_dists, 'fg': fg_nonparam}
punt_fg_probs_df = pd.DataFrame(punt_fg_probs)

# send fg prob and punt distance calculations to csv to access in R
# punt_fg_probs_df.to_csv(r'/Users/EthanLee/nfl-runpass/punt_fg_probs.csv')

## find the probability of success for run and pass plays, as well as yards gained on failed plays and interception probabilities

def failed_yds(play_type, max_ytg = 21):

    plays = nfl_pbp[nfl_pbp.play_type == play_type]

    # probability of 1st down given yards to go
    firstDown_prob = np.zeros(max_ytg-1)
    for ytg in range(1, max_ytg):
        firstDown_prob[ytg-1] = np.mean(plays.yards_gained[plays.ydstogo == ytg] >= ytg)

    # piecewise polynomial smooth
    firstDown_poly = poly_smooth(np.arange(1,max_ytg), firstDown_prob.copy(), deg=3)[0]

    if play_type == 'run':
        run_probs_df = pd.DataFrame(firstDown_poly)
        # send run prob calculations to csv to access in R
        # run_probs_df.to_csv(r'/Users/EthanLee/nfl-runpass/run_probs.csv')
    else:
        pass_probs_df = pd.DataFrame(firstDown_poly)
        # send pass prob calculations to csv to access in R
        # pass_probs_df.to_csv(r'/Users/EthanLee/nfl-runpass/pass_probs.csv')

    if play_type == 'pass':
        # probability of interception at each yardline, and average yards gained
        intercept_probs = np.zeros(99) # probability of interception (not pick 6)
        intercept_td_probs = np.zeros(99) # probability of pick 6
        intercept_returns = np.zeros(99) # yards gained in return
        for i in range(99):
            ydline = plays[(plays.yardline_100 == i+1)]
            intercept_td_probs[i] = np.mean((ydline.interception == 1) & (ydline.touchdown == 1))
            intercept_probs[i] = np.mean((ydline.interception == 1) & (ydline.touchdown == 0))
            if sum(ydline.interception == 1) == 0:
                intercept_returns[i] = 0
            else:
                intercept_returns[i] = np.mean(ydline[(ydline.interception == 1)].air_yards - ydline[(ydline.interception == 1)].return_yards)
        # non-parametric smooth
        intercept_prob_nonparam = nonparam_smooth(intercept_probs.copy(), window=21)
        intercept_prob_nonparam[intercept_prob_nonparam < 0] = 0
        intercept_prob_nonparam[intercept_prob_nonparam > 1] = 1
        intercept_td_prob_nonparam = nonparam_smooth(intercept_td_probs.copy(), window=21)
        intercept_td_prob_nonparam[intercept_td_prob_nonparam < 0] = 0
        intercept_td_prob_nonparam[intercept_td_prob_nonparam > 1] = 1
        intercept_return_nonparam = nonparam_smooth(intercept_returns.copy(), window=21)

        intercept_probs = {'prob': intercept_prob_nonparam, 'td_prob': intercept_td_prob_nonparam, 'return': intercept_return_nonparam}
        intercept_probs_df = pd.DataFrame(intercept_probs)

        # send interception calculations to csv to access in R
        # intercept_probs_df.to_csv(r'/Users/EthanLee/nfl-runpass/intercept_probs.csv')

    # calculate average yards gained on failed play_type, given yardline and yards to go
    avg_fail_yds = {}
    for yardline in range(1, 100):
        avg_fail_yds_yardline = np.zeros(max_ytg-1)
        for ytg in range(1, max_ytg):
            if yardline >= ytg and yardline-ytg < 90:
                avg_yds = np.mean(plays[((plays.yardline_100 == yardline) & (plays.ydstogo == ytg) & (plays.yards_gained < ytg))].yards_gained)
                if avg_yds != avg_yds: # avg_yards is NaN
                    print("NAN for", play_type, "play at yardline", yardline, "ytg", ytg)
                    avg_yds = 0
            avg_fail_yds_yardline[ytg-1] = avg_yds
        avg_yds_poly = poly_smooth(np.arange(1,max_ytg), avg_fail_yds_yardline.copy(), deg=3)[0]
        avg_fail_yds[yardline] = avg_yds_poly
    return avg_fail_yds

run_play = failed_yds('run')
pass_play = failed_yds('pass')

## Turn run_play into CSV to access in R
run_df = pd.DataFrame(run_play)
# run_df.to_csv(r'/Users/EthanLee/nfl-runpass/run_fails.csv')

## Turn pass_play into CSV to access in R
pass_df = pd.DataFrame(pass_play)
# pass_df.to_csv(r'/Users/EthanLee/nfl-runpass/pass_fails.csv')
