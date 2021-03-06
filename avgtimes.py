import pandas as pd
import numpy as np
import scipy.signal as signal

def nonparam_smooth(y,smooth_type='savgol',window=19):
    if smooth_type=='savgol':
        y[~np.isnan(y)] = signal.savgol_filter(y[~np.isnan(y)],window,2)
    return y

# load data
DATADIR = '/Users/EthanLee/Desktop/STAT 143/'
nfl = pd.read_csv(DATADIR + 'NFL_PbP_2009_2018_4thDownAnalysis.csv')

# Get time of play for pass and run plays
time_of_play = []
for index, row in nfl.iterrows():
    if index != (nfl.shape[0]-1) and (nfl.iloc[index].play_type == "pass" or nfl.iloc[index].play_type == "run"):
        if(nfl.iloc[index].game_seconds_remaining >= nfl.iloc[index+1].game_seconds_remaining):
            time_of_play.append(nfl.iloc[index].game_seconds_remaining - nfl.iloc[index+1].game_seconds_remaining)
        else:
            # random value that we won't ever use
            time_of_play.append(-10)
    else:
        # random value that we won't ever use
        time_of_play.append(-10)

nfl["time_of_play"] = time_of_play

#Get only 3rd and 4th down data
nfl_pbp = nfl[nfl['down'] >=3]

### Average time of run, pass

# getting average time of run plays
run_plays = nfl_pbp[nfl_pbp['play_type']=='run']
run_play_time = np.zeros(15)
for i in range(15):
    rows = run_plays.ydstogo==i+1
    temp = run_plays[rows]
    run_play_time[i] = np.mean(temp[temp["time_of_play"]>=0].time_of_play)
    if run_play_time[i] != run_play_time[i]: # if nan
        run_play_time[i] = 0

# nonparametric smooth
runtime_nonparam = nonparam_smooth(run_play_time.copy(), window=13)

runtime_df = pd.DataFrame(runtime_nonparam)
# send run play time calculations to csv to access in R
runtime_df.to_csv(r'/Users/EthanLee/nfl-runpass/runtime.csv')

# getting average time of pass plays
pass_plays = nfl_pbp[nfl_pbp['play_type']=='pass']
pass_play_time = np.zeros(20)
for i in range(20):
    rows = pass_plays.ydstogo==i+1
    temp = pass_plays[rows]
    pass_play_time[i] = np.mean(temp[temp["time_of_play"]>=0].time_of_play)
    if pass_play_time[i] != pass_play_time[i]: # if nan
        pass_play_time[i] = 0

# nonparametric smooth
passtime_nonparam = nonparam_smooth(pass_play_time.copy(), window=13)

passtime_df = pd.DataFrame(passtime_nonparam)
# send pass play time calculations to csv to access in R
passtime_df.to_csv(r'/Users/EthanLee/nfl-runpass/passtime.csv')
