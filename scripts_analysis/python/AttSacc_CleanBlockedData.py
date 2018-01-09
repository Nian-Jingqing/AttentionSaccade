# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:05:12 2018

@author: sammirc

purpose of this script is to clean the blocked data (continuous signal, better for artefact removal) before subsequently epoching into trials
"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import re
import os
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade' #laptop directory
os.chdir(workingfolder)
import myfunctions

%matplotlib
np.set_printoptions(suppress = True)

import multiprocessing
n_cores = multiprocessing.cpu_count() #8 cores on workstation

#%%

eyedat        = os.path.join(workingfolder, 'eyes')
sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

parts   = ['a','b']

#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify

epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
hrefsubs     = [7, 8]
lowsamp_subs = [4]

list_fnames = sorted(os.listdir(os.path.join(eyedat, 'blocked_data')))
#%%
for fid in range(len(list_fnames)):
    print 'working on file %02d/%02d' %(fid+1, len(list_fnames))
#%%
fid = 0
#load in the blocked data    
fname = list_fnames[fid]
print '\n loading blocked data from pickle'
with open(os.path.join(eyedat, 'blocked_data', fname), 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'

# ds contains the data for this file. it is a list of dictionaries len(ds) = nblocks for the file
# keys for each block dictionary are:

# Msg           - contains all the messages sent in this block (triggers for each trial, etc)
# Sfix          - start sample points for fixations (in tracker time, using SR research methods)
# Efix          - event coded version of fixations
# Sblk          - start sample points for blinks (in tracker time, using SR research methods)
# Eblk          - event coded version of blinks
# Ssac          - start sample points of saccades (in tracker time, using SR research methods)
# Esac          - event coded version of saccades (SR research methods)
# trackertime   - timepoints in tracker time
# lx            - time-series for the left x value
# ly            - time-series for the left y value
# lp            - time-series for the left pupil value
# rx            - time-series for the right x value
# ry            - time-series for the right y value
# rp            - time-series for the right pupil value
# av_x          - time-series averaged (pre-cleaning) x value across eyes     (this is probably noisier than necessary as it's from raw data)
# av_y          - time-series averaged (pre-cleaning) y value across eyes     (this is probably noisier than necessary as it's from raw data)
# av_p          - time-series averaged (pre-cleaning) pupil value across eyes (this is probably noisier than necessary as it's from raw data)



# perform artefact removal on the blocked signals, and show figures verifying quality of the signal removal




#%%

blinks = copy.deepcopy(ds[0]['Eblk'])
blink = [x for x in range(len(blinks)) if len(blinks[x]) > 0]
blinks = [blinks[x] for x in blink]
blinks = [x for y in blinks for x in y] #flatten the nested lists into one list

plt.figure()
#plt.plot(ds[0]['trackertime'], ds[0]['lx'], color = '#252525', label = 'left x' )
plt.plot(ds[0]['trackertime'], ds[0]['rx'], color = '#08519c' , label = 'right x'); plt.legend()
for blink in blinks:
    if blink[1] == 'R':
        plt.axvline(blink[2], ls = 'dashed', color = '#006d2c') #dark green start
        #plt.axvline(blink[3], ls = 'dashed', color = '#31a354') #pale green end
    #elif blink[1] == 'L':
    #    plt.axvline(blink[2], ls = 'dashed', color = '#e6550d') #dark red/orange start
        #plt.axvline(blink[3], ls = 'dashed', color = '#fd8d3c') #pale red/orange end        


#%%
#exploring

for i in range(len(blinks)):
    blink_id = i
    
    blink = blinks[blink_id - 1]
    blinks[0]
    ds[0]['trackertime'][0]
    int(blink[2]) #start trackertime
    int(blink[3]) #end trackertime
    
    start = np.where(ds[0]['trackertime'] == int(blink[2]))[0][0]
    end   = np.where(ds[0]['trackertime'] == int(blink[3]))[0][0] 
    epoch = copy.deepcopy(ds[0]['lx'][start-500:end+500])
    
    
    plt.figure()
    plt.plot(ds[0]['trackertime'][start-500:end+500], epoch)
    plt.axvline(blink[2], color = 'grey', ls = 'dashed')
    plt.axvline(int(blink[2])-40, color = '#3182bd', ls = '--')
    plt.axvline(blink[3], color = 'grey', ls = 'dashed')
    plt.axvline(int(blink[3])+40, color = '#ca0020', ls = '--')

"""basically if you plot these figures, you see one case where the output
from the eyelink says there's a blink, but it started substantially earlier
this means that the blink detection from SR research is going to be a little
unreliable, and it's best to generate your own code to do this

the best way of doing this is probably just identifying series of missing
samples in the data (absence of value = blink) and creating a log of them.
then probably plot these and see if better.


to be fair though, credit to them, it's mostly ok'
"""
temp = copy.deepcopy(ds[0]) #create a new, temporary variable for one block of data
#%%

mlx = np.array(np.isnan(temp['lx']) == True, dtype = int)
dlx = np.diff(mlx)
s_lx = np.squeeze(np.where(dlx ==  1))
e_lx = np.squeeze(np.where(dlx == -1))


mly = np.array(np.isnan(temp['ly']) == True, dtype = int)
dly = np.diff(mly)
s_ly = np.squeeze(np.where(dly ==  1))
e_ly = np.squeeze(np.where(dly == -1))

mrx = np.array(np.isnan(temp['rx']) == True, dtype = int)
drx = np.diff(mrx)
s_rx = np.squeeze(np.where(drx ==  1))
e_rx = np.squeeze(np.where(drx == -1))

mry = np.array(np.isnan(temp['ry']) == True, dtype = int)
dry = np.diff(mry)
s_ry = np.squeeze(np.where(dry ==  1))
e_ry = np.squeeze(np.where(dry == -1))

mx  = np.array((mlx+mrx) == 2, dtype = int)
dx  = np.diff(mx)
s_x = np.squeeze(np.where(dx ==  1))
e_x = np.squeeze(np.where(dx == -1))

#the left x and left y should be missing the same periods of data
#right x and right y should be missing the same periods of data

#turn these data into blink events for right and left eyes

Eblk_lx = []; Eblk_ly = []
Eblk_rx = []; Eblk_ry = []

#left x
for i in range(len(s_lx)):
    start = s_lx[i]
    if i < len(e_lx): #check within range of the number of missing periods in data
        end = e_lx[i]
    elif i == len(e_lx):
        end = e_lx[-1]
    else:
        end = e_lx[-1]
    ttime_start = temp['trackertime'][start]
    ttime_end   = temp['trackertime'][end]
    dur = end-start
    #create a blink event structure:
    # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
    evnt = ['LX_BLK', start, end, ttime_start, ttime_end, dur]
    Eblk_lx.append(evnt)

#left y
for i in range(len(s_ly)):
    start = s_ly[i]
    if i < len(e_ly): #check within range of the number of missing periods in data
        end = e_ly[i]
    elif i == len(e_ly):
        end = e_ly[-1]
    else:
        end = e_ly[-1]
    ttime_start = temp['trackertime'][start]
    ttime_end   = temp['trackertime'][end]
    dur = end-start
    #create a blink event structure:
    # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
    evnt = ['LY_BLK', start, end, ttime_start, ttime_end, dur]
    Eblk_ly.append(evnt)

#right x
for i in range(len(s_rx)):
    start = s_rx[i]
    if i < len(e_rx): #check within range of the number of missing periods in data
        end = e_rx[i]
    elif i == len(e_rx):
        end = e_rx[-1]
    else:
        end = e_rx[-1]
    ttime_start = temp['trackertime'][start]
    ttime_end   = temp['trackertime'][end]
    dur = end-start
    #create a blink event structure:
    # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
    evnt = ['RX_BLK', start, end, ttime_start, ttime_end, dur]
    Eblk_rx.append(evnt)

#right y
for i in range(len(s_ry)):
    start = s_ry[i]
    if i < len(e_ry): #check within range of the number of missing periods in data
        end = e_ry[i]
    elif i == len(e_ry):
        end = e_ry[-1]
    else:
        end = e_ry[-1]
    ttime_start = temp['trackertime'][start]
    ttime_end   = temp['trackertime'][end]
    dur = end-start
    #create a blink event structure:
    # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
    evnt = ['RY_BLK', start, end, ttime_start, ttime_end, dur]
    Eblk_ry.append(evnt)





#quickly just check this section to see how accurate
#plt.figure()
#plt.plot(temp['trackertime'], temp['lx'], ls = '-')
#for i in range(len(Eblk_lx)):
#    plt.axvline(Eblk_lx[i][3], ls = '--', color = '#636363')
#    plt.axvline(Eblk_lx[i][4], ls = '--', color = '#636363')
#%%





plt.figure()
plt.plot(ds[0]['trackertime'], ds[0]['lx'], color = 'green')
plt.title('left x time-series for blocked data, triggers are blinks')
for i in range(len(ds[0]['Eblk'])):
    if len(ds[0]['Eblk'][i]) >0:
        if len(ds[0]['Eblk'][i]) == 1:
            blink = ds[0]['Eblk'][i][0]
        if ds[0]['Eblk'][i][0][1] == 'L':
            start = ds[0]['Eblk'][i][0][2]
            end   = ds[0]['Eblk'][i][0][3]
            plt.axvline(start, ls = 'dashed', color = 'red')
            plt.axvline(end, ls = 'dashed', color = 'grey')
    elif len(ds[0]['Eblk'])










plt.figure()
plt.plot(blocked_data[0]['trackertime'], blocked_data[0]['lx'], color = 'green')
plt.plot(blocked_data[0]['trackertime'], blocked_data[0]['rx'], color = 'blue')
plt.title('trial eye-x data across whole of block 1, triggers are blinks')
for i in range(len(blocked_data[0]['Sblk'])):
    if len(blocked_data[0]['Sblk'][i]) > 0:
        for blink in range(len(blocked_data[0]['Sblk'][i])):
            plt.axvline(int(blocked_data[0]['Sblk'][i][blink][2]), ls = 'dashed', color = 'red')



















