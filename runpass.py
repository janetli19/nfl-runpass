#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns


# smoothing functions

def nonparam_smooth(y,smooth_type='savgol',window=21):
    if smooth_type=='savgol':
        y[~np.isnan(y)] = signal.savgol_filter(y[~np.isnan(y)],window,2)
    return y

def poly_smooth(x,y,deg):
    nanvals = np.isnan(y)
    pfit,cov = np.polyfit(x[~nanvals],y[~nanvals],deg, cov=True)
    yp = np.poly1d(pfit)(x)
    return yp,pfit,cov


def get_decisions(nfl_pbp, max_ytg=21):
    """
    Returns matrix of optimal decision and matrix of expected points where row = yardline and col = yards to go
    For decision matrix: 0 = impossible, 1 = field goal, 2 = punt, 3 = run, 4 = pass
    For expected points matrix: 300 = impossible

    nfl_pbp: NFL play-by-play dataset
    max_ytg: max yards to go + 1
    """

    ### Going for it on 4th down
    
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

    def get_firstdown(play_type):
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
    for yds in range(1, 100):
        punt_dist = np.mean(plays[plays.yardline_100==yds].kick_distance) - np.mean(plays[plays.yardline_100==yds].return_yards)
        if punt_dist != punt_dist: # if nan, then touchback
            punt[yds] = -1 * firstDown_pts[80]
        else:
            punt[yds] = -1 * firstDown_pts[min(round(100-yds+punt_dist), 80)]


    ### Decision and expected points matrices
    decision = np.zeros((max_ytg, 100))
    points = np.zeros((max_ytg, 100))
    points.fill(300)
    for yardline in range(1,100):
        for ytg in range(1, min(yardline+1, max_ytg)):
            if ytg > yardline:
                decision[ytg][yardline] = 0
                points[ytg][yardline] = 300
            else:
                decision[ytg][yardline] = np.array([fg[yardline], punt[yardline], run_play[yardline][ytg], pass_play[yardline][ytg]]).argmax()+1
                points[ytg][yardline] = max([fg[yardline], punt[yardline], run_play[yardline][ytg], pass_play[yardline][ytg]])
    return decision, points

def get_coach_decisions(nfl_pbp, max_ytg=21):
    """
    Returns matrix of most popular coaches' decisions where row = yardline and col = yards to go
    0 = impossible, 1 = field goal, 2 = punt, 3 = run, 4 = pass

    nfl_pbp: NFL play-by-play dataset
    max_ytg: max yards to go + 1
    """    
    decision = np.zeros((max_ytg, 100))
    for yardline in range(1,100):
        for ytg in range(1, min(yardline+1, max_ytg)):
            if ytg > yardline:
                decision[ytg][yardline] = 0
            else:
                plays = nfl_pbp[((nfl_pbp.down == 4) & (nfl_pbp.yardline_100 == yardline) & (nfl_pbp.ydstogo == ytg))]
                if len(plays) != 0:
                    decision[ytg][yardline] = 1 + np.nanargmax([
                        np.sum(plays.field_goal_attempt == 1),
                        np.sum(plays.play_type == 'punt'),
                        np.sum(plays.play_type == 'run'),
                        np.sum(plays.play_type == 'pass')
                    ])
                else: 
                    decision[ytg][yardline] = 0
    return decision

### Plot decisions

def plot_decisions(decision, max_ytg=21, title='', filename='decisions.png'):
    plt.figure(figsize=(20,5))
    with sns.axes_style('white'):
        ax = sns.heatmap(decision, mask=np.ma.masked_values(decision, 0).mask, vmin=0, vmax=4, xticklabels=5, 
                    yticklabels=5, cmap='viridis', cbar=True, linewidths=.1)
        ax.set(xlim=(1,100), ylim=(max_ytg,1))
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Field position (yards to goal)", fontsize=15)
    plt.ylabel("4th and ...", fontsize=15)
    plt.savefig(filename)


### Plot expected points

def plot_points(points, max_ytg=21, title=''):
    plt.figure(figsize=(20,5))
    with sns.axes_style('white'):
        ax = sns.heatmap(points, mask=np.ma.masked_values(points, 300).mask, xticklabels=5, yticklabels=5, 
                        cmap='RdYlGn', cbar=True, linewidths=.1)
        ax.set(xlim=(1,100), ylim=(max_ytg,1))
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Field position (yards to goal)", fontsize=15)
    plt.ylabel("4th and ...", fontsize=15)
    plt.savefig('points.png')


# load data
DATADIR = ''
full_data = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

# nfl_pbp = full_data[full_data.qtr == 1]
nfl_pbp = full_data

# plot coaches' decisions
coaches = get_coach_decisions(nfl_pbp)
plot_decisions(coaches, title="Coaches' Decisions on 4th Down", filename='coaches.png')
# plot recommended decisions
decision, points = get_decisions(nfl_pbp)
plot_decisions(decisions, title='Decision')
plot_points(points, title='Expected Value (Points) of First Down')