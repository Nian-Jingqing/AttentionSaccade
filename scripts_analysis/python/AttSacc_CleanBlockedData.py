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
#workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade' #laptop directory
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

    #load in the blocked data    
    fname = list_fnames[fid]
    print '\n loading blocked data from pickle'
    with open(os.path.join(eyedat, 'blocked_data', fname), 'rb') as handle:
        ds = cPickle.load(handle)
    print 'finished loading data'
    
    # ds contains the data for this file. it is an array of dictionaries len(ds) = nblocks for the file
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
    
    ds = myfunctions.find_missing_periods(dataset,24)

#%%
dataset = myfunctions.find_missing_periods(dataset,24)
block1  = myfunctions.interpolateBlinks_Blocked(dataset[0], 'lx')


#%%
# perform artefact removal on the blocked signals, and show figures verifying quality of the signal removal
#%%
#exploring the blinks to highlight issues

blinks = copy.deepcopy(ds[0]['Eblk'])
blink = [x for x in range(len(blinks)) if len(blinks[x]) > 0]
blinks = [blinks[x] for x in blink]
blinks = [x for y in blinks for x in y] #flatten the nested lists into one list

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
    plt.axvline(blink[2], color = '#bdbdbd', ls = 'dashed')
    plt.axvline(blink[3], color = '#bdbdbd', ls = 'dashed')
    plt.title('blink detection using SR research output')


    
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
#%% find missing periods relating to blinks, in each separate trace and add an event structure to the data
"""
don't need to do this for pupil as for each trace in gaze we should also interpolate appropriate areas in
the pupil trace of the appropriate eye """

for block in range(len(ds)): #loop through all blocks of the data
    s_lx = []; s_rx = [];
    s_ly = []; s_ry = [];
    e_lx = []; e_rx = [];
    e_ly = []; e_ry = [];

    #find missing data in each gaze trace (left x & y, right x & y) separately for precision
    mlx = np.array(np.isnan(ds[block]['lx']) == True, dtype = int) # array of 1's and 0's for if data is missing at that time point
    dlx = np.diff(mlx) #find change-points, +1 means goes from present to absent, -1 from absent to present
    s_lx = np.squeeze(np.where(dlx ==  1)) #find points where it starts to be missing
    e_lx = np.squeeze(np.where(dlx == -1))+1 #find points where it stops being missing
    
    mly = np.array(np.isnan(ds[block]['ly']) == True, dtype = int)
    dly = np.diff(mly)
    s_ly = np.squeeze(np.where(dly ==  1))
    e_ly = np.squeeze(np.where(dly == -1))+1
    
    mrx = np.array(np.isnan(ds[block]['rx']) == True, dtype = int)
    drx = np.diff(mrx)
    s_rx = np.squeeze(np.where(drx ==  1))
    e_rx = np.squeeze(np.where(drx == -1))+1
    
    mry = np.array(np.isnan(ds[block]['ry']) == True, dtype = int)
    dry = np.diff(mry)
    s_ry = np.squeeze(np.where(dry ==  1))
    e_ry = np.squeeze(np.where(dry == -1))+1

    Eblk_lx = []; Eblk_ly = []
    Eblk_rx = []; Eblk_ry = []
    #left x
    for i in range(s_lx.size):
        if s_lx.size == 1:
            start = s_lx.tolist()
            end   = e_lx.tolist()
        else:
            start = s_lx[i]
            if i < e_lx.size: #check within range of the number of missing periods in data
                end = e_lx[i]
            elif i == e_lx.size:
                end = e_lx[-1]
            else:
                end = e_lx[-1]
        ttime_start = ds[block]['trackertime'][start]
        ttime_end   = ds[block]['trackertime'][end]
        dur = end-start
            #create a blink event structure:
            # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
        evnt = ['LX_BLK', start, end, ttime_start, ttime_end, dur]
        Eblk_lx.append(evnt)
    #left y
    for i in range(s_ly.size):
        if s_ly.size == 1:
            start = s_ly.tolist()
            end   = e_ly.tolist()
        else:
            if i < e_ly.size: #check within range of the number of missing periods in data
                end = e_ly[i]
            elif i == e_ly.size:
                end = e_ly[-1]
            else:
                end = e_ly[-1]
        ttime_start = ds[block]['trackertime'][start]
        ttime_end   = ds[block]['trackertime'][end]
        dur = end-start
        #create a blink event structure:
        # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
        evnt = ['LY_BLK', start, end, ttime_start, ttime_end, dur]
        Eblk_ly.append(evnt)   
    #right x
    for i in range(s_rx.size):
        if s_rx.size == 1:
            start = s_rx.tolist()
            end   = e_rx.tolist()
        else:
            if i < e_rx.size: #check within range of the number of missing periods in data
                end = e_rx[i]
            elif i == e_rx.size:
                end = e_rx[-1]
            else:
                end = e_rx[-1]
        ttime_start = ds[block]['trackertime'][start]
        ttime_end   = ds[block]['trackertime'][end]
        dur = end-start
        #create a blink event structure:
        # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
        evnt = ['RX_BLK', start, end, ttime_start, ttime_end, dur]
        Eblk_rx.append(evnt)    
    #right y
    for i in range(s_ry.size):
        if s_ry.size == 1:
            start = s_ry.tolist()
            end   = e_ry.tolist()
        else:
            if i < e_ry.size: #check within range of the number of missing periods in data
                end = e_ry[i]
            elif i == e_ry.size:
                end = e_ry[-1]
            else:
                end = e_ry[-1]
        ttime_start = ds[block]['trackertime'][start]
        ttime_end   = ds[block]['trackertime'][end]
        dur = end-start
        #create a blink event structure:
        # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
        evnt = ['RY_BLK', start, end, ttime_start, ttime_end, dur]
        Eblk_ry.append(evnt)
    #append these new structures to the dataset
    ds[block]['Eblk_lx'] = Eblk_lx
    ds[block]['Eblk_rx'] = Eblk_rx
    ds[block]['Eblk_ly'] = Eblk_ly
    ds[block]['Eblk_ry'] = Eblk_ry


#%% quick visualisation to see if this is better now
block = 5
plt.figure()
plt.plot(ds[block]['trackertime'], ds[block]['lx'], label = 'lx'); plt.legend()
for blink in ds[block]['Eblk_lx']:
    #if blink[-1] < 20:
    #    plt.axvline(blink[3], ls = '--', color = '#41ab5d') 
    if blink[-1] in range(50,101):
        plt.axvline(blink[3], ls = '--', color = '#3182bd')
        plt.axvline(blink[4], ls = '--', color = '#636363')
#    elif blink[-1] > 40:
#        plt.axvline(blink[3], ls = '--', color = '#fc4e2a')
#        plt.axvline(blink[4], ls = '--', color = '#fc4e2a')
        
#%%
reload(myfunctions)
block = 5

data = copy.deepcopy(ds[block])
dat = copy.deepcopy(ds[block])
dat = myfunctions.interpolateBlinks_Blocked(dat, 'lx')



plt.figure()
plt.plot(dat['trackertime'] , dat['lx'] , label = 'cleaned', color = '#31a354', lw=1)
plt.plot(data['trackertime'], data['lx'], label = 'raw', color = '#3182bd', lw=1)

#%% generate some figures to see how effective the cubic spline is for different missing period durations
        
        
block           = 5
data            = copy.deepcopy(ds[block])
lx              = copy.deepcopy(data['lx'])
blinks          = np.array(copy.deepcopy(data['Eblk_lx']))
uniques         = np.array(np.unique(blinks[:,5]).tolist(), dtype = int) #unique durations of missing periods

blinks          = blinks[:,1:].astype(float).astype(int)
short_dur_inds  = np.where(np.in1d(blinks[:,4], range(21)))[0]
medium_dur_inds = np.where(np.in1d(blinks[:,4], range(21,51)))[0]
long_dur_inds   = np.where(blinks[:,4] > 50)[0]

short_blinks = blinks[short_dur_inds,:]
medium_blinks = blinks[medium_dur_inds,:]
long_blinks = blinks[long_dur_inds,:]

#for blink in medium_blinks:
#    start, end = blink[0], blink[1]
#    epoch = data['lx'][start-500:end+500]
#    plt.figure()
#    plt.plot(data['trackertime'][start-500:end+500], epoch)
#    plt.axvline(blink[2], color = '#bdbdbd', ls = 'dashed')
#    plt.axvline(blink[3], color = '#bdbdbd', ls = 'dashed')
#    plt.title('missing period between 20-50 samples(ms)')

#interpolate across the smaller missing periods before doing the long ones
for blink in short_blinks:
    start, end               = blink[0], blink[1] #get start and end periods of the missing data
    to_interp                = np.arange(start, end) #get all time points to be interpolated over
    
    #set up linear interpolation
    inttime                  = np.array([start,end])
    inttrace                 = np.array([lx[start], lx[end]])
    fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')
    
    interptrace              = fx_lin(to_interp)
    lx[to_interp] = interptrace
for blink in medium_blinks:
    start, end               = blink[0], blink[1] #get start and end periods of the missing data
    to_interp                = np.arange(start, end) #get all time points to be interpolated over
    
    #set up linear interpolation
    inttime                  = np.array([start,end])
    inttrace                 = np.array([lx[start], lx[end]])
    fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')
    
    interptrace              = fx_lin(to_interp)
    lx[to_interp] = interptrace


raw = copy.deepcopy(lx)
#show cubic spline interpolation over these longer ones, see how it works
count = 1
for blink in long_blinks:
    start, end = blink[0], blink[1]
    epoch      = raw[start-500:end+500]
    
    cleaned    = copy.deepcopy(lx)
    window     = [start-80, start-40, end+40, end+80]
    inttime    = np.array(window)
    intlx      = np.array([np.nanmedian(lx[start-80:start-40]), np.nanmedian(lx[start-40:start-1]),
                           np.nanmedian(lx[end+1:end+40])     , np.nanmedian(lx[end+40:end+80]) ])
    fx_cub     = sp.interpolate.interp1d(inttime, intlx, kind = 'cubic')
    
    to_interp  = np.arange(start-20, end+20)
    cleaned[to_interp] = fx_cub(to_interp)
    lx[to_interp] = fx_cub(to_interp)
    
    if count in [69, 75, 83, 85, 89, 101]:
        plt.figure()
        plt.plot(data['trackertime'][start-500:end+500], epoch, label = 'raw trace', color = '#3182bd', lw=1)
        plt.plot(data['trackertime'][start-500:end+500], cleaned[start-500:end+500], label = 'cubic spline interpolated blink', ls = '--', color = '#31a354', lw=1)
        plt.axvline(blink[2], color = '#bdbdbd', ls = 'dashed')
        plt.axvline(blink[3], color = '#bdbdbd', ls = 'dashed')
        plt.title('longer period above 50 samples(ms)')
        plt.legend()
        plt.savefig('/home/sammirc/Experiments/Nick/AttentionSaccade/brognitionday_18_figures/long_length_missing%02d.eps'%(count),dpi=300)
    count += 1


# looks like the right sequence is to linear interpolate the small missing periods
# then cubic spline the big ones. This doesn't work on all of them, due to sequential, nearby sections having large missing patches
# the same routine for missing period detection should then be run to re-identify them
# and probably just linear interpolate over these as it's clear that a cubic spline won't fit

plt.close('all')

#%%


for i in range(len(blinks)):
    blink_id = i
    blink = 

    start = np.where(ds[0]['trackertime'] == int(blink[2]))[0][0]
    end   = np.where(ds[0]['trackertime'] == int(blink[3]))[0][0] 
    epoch = copy.deepcopy(ds[0]['lx'][start-500:end+500])
    
    cleaned         = copy.deepcopy(epoch)
    window          = [start-80, start-40, end+40, end+80]
    instart, intend = start-40, end+40
    inttime         = np.array(window)
    intblk          = np.array([ds[0]['lx'][window[0]], ds[0]['lx'][window[1]],ds[0]['lx'][window[2]],ds[0]['lx'][window[3]]])
    fx_cub
    
    plt.figure()
    plt.plot(ds[0]['trackertime'][start-500:end+500], epoch)
    plt.axvline(blink[2], color = '#bdbdbd', ls = 'dashed')
    plt.axvline(int(blink[2])-40, color = '#31a354', ls = '--')
    plt.axvline(int(blink[2])-80, color = '#de2d26', ls = '--')
    plt.axvline(blink[3], color = '#bdbdbd', ls = 'dashed')
    plt.axvline(int(blink[3])+40, color = '#31a354', ls = '--')
    plt.axvline(int(blink[3])+80, color = '#de2d26', ls = '--')
    
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

# in my interpolation function, it takes start-80, start-40, end+40, end+80ms around the blink
# just plot some figures to explore whether this is still sufficient. can possible extend this now using the continuous data.
# one thing to maybe think about is a pca approach to removing them. this could be a good idea, and maybe remove less of the signal around it
from scipy import interpolate

temp = copy.deepcopy(ds[block])
temp['blocktime'] = temp['trackertime'] - temp['trackertime'][0]

lx     = copy.deepcopy(temp['lx'])
blinks = np.array(copy.deepcopy(temp['Eblk_lx']))
uniques = np.array(np.unique(blinks[:,5]).tolist(), dtype = int)


#if the duration of a missing period is under 15 samples, then take the start and end points
#(and 10ms either side) and linear interpolate across the break.

#%%
blinks = blinks[:,1:].astype(float).astype(int)
short_dur_inds = np.where(blinks[:,4] in range(20,50))[0]

#%%
blink = blinks[short_dur_inds[9]].tolist() # 8ms missing period
start, end       = blink[0], blink[1]
start_tt, end_tt = blink[2], blink[3]


# set up interpolation and assess methods
intstart, intend = start - 5, end + 5 #take 20 sample (ms) window around the start/end of missing
#to_interp = np.arange(intstart, intend)
to_interp = np.arange(start, end)
#inttime = np.array([intstart, intend])
inttime = np.array([start, end])
#intlx = np.array([lx[instart], lx[intend]])
intlx = np.array([lx[start], lx[end]])

#fx_lin = sp.interpolate.interp1d(inttime, intlx, kind = 'linear')
fx_lin = sp.interpolate.interp1d(inttime, intlx, kind = 'linear')

lininterp_lx = fx_lin(to_interp)
lin_lx = copy.deepcopy(lx)
lin_lx[to_interp] = lininterp_lx



plt.figure()
plt.plot(temp['trackertime'][start-100:end + 100], temp['lx'][start-100:end + 100], label = 'raw',                       color = '#386cb0')
plt.plot(temp['trackertime'][start-100:end + 100], lin_lx[start - 100:end + 100],   label = 'linear interpolated',       color = '#fdc086')
plt.axvline(start_tt, ls = '--', color = '#999999')
plt.axvline(end_tt, ls = '--', color = '#999999')
plt.title('linear interpolation over 8ms missing data patch')
plt.legend()




#%%
blink = blinks[1].tolist()
start, end       = int(blink[1]), int(blink[2])
start_tt, end_tt = int(float(blink[3])), int(float(blink[4])) 


# set up interpolation and assess methods
intstart, intend = start - 20, end + 20 #take 20 sample (ms) window around the start/end of missing
to_interp = np.arange(start-10, end + 10)
inttime = np.array([intstart, intend])
intlx = np.array([lx[instart], lx[intend]])

fx_lin = sp.interpolate.interp1d(inttime, intlx, kind = 'linear')

intstart_cub, intend_cub = start-60, end + 60
to_interp_cub = np.arange(start-20, end + 20)
inttime_cub   = np.array([intstart_cub, start-20,end+20, intend_cub])
intlx_cub     = np.array([lx[intstart_cub], lx[start-20], lx[end+20], lx[intend_cub]])
fx_cub        = sp.interpolate.interp1d(inttime_cub, intlx_cub, kind = 'cubic')


lininterp_lx = fx_lin(to_interp)
lin_lx = copy.deepcopy(lx)
lin_lx[to_interp] = lininterp_lx

cubinterp_lx = fx_cub(to_interp_cub)
cub_lx = copy.deepcopy(lx)
cub_lx[to_interp_cub] = cubinterp_lx


plt.figure()
plt.plot(temp['trackertime'][start-100:end + 100], temp['lx'][start-100:end + 100], label = 'raw',                       color = '#386cb0')
plt.plot(temp['trackertime'][start-100:end + 100], lin_lx[start - 100:end + 100],   label = 'linear interpolated',       color = '#fdc086')
plt.plot(temp['trackertime'][start-100:end + 100], cub_lx[start - 100:end + 100],   label = 'cubic spline interpolated', color = '#7fc97f')
plt.axvline(start_tt, ls = '--', color = '#999999')
plt.axvline(end_tt, ls = '--', color = '#999999')
plt.axvline(start_tt - 20, ls = '--', color = '#bdbdbd', label = '20ms window')
plt.axvline(end_tt   + 20, ls = '--', color = '#bdbdbd', label = '20ms window')
plt.axvline(start_tt - 60, ls = '--', color = '#404040', label = '60ms window')
plt.axvline(end_tt   + 60, ls = '--', color = '#404040', label = '60ms window')
plt.title('methods of interpolation over 101ms missing data patch')
plt.legend()

