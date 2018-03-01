#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:59:40 2018

@author: sammirc
"""

import numpy as np
import pandas as pd
import os.path as op
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import copy
import cPickle
import glob
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
#workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade' #laptop directory

os.chdir(workingfolder)
import BCEyes

np.set_printoptions(suppress = True)

behaviour     = op.join(workingfolder, 'behaviour', 'csv')
eyedir        = op.join(workingfolder, 'eyes')
cleaned_dir   = op.join(eyedir, 'gaze_cleaned')


os.chdir(behaviour)
list_fnames = sorted(glob.glob('*.csv')) #list only .csv files for behavioural data
os.chdir(workingfolder)
#%%

fid = list_fnames[6]
fid_sub = fid.split('.')[0]

beh_fname = op.join(behaviour, fid)
eye_fname = op.join(cleaned_dir, fid_sub+'_parsed_cleanedGaze.pickle')


print '\nloading parsed data from pickle'
with open(eye_fname, 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'

#before you can run the loop below, need to know which blocks are saccade blocks otherwise it will fail with epoching as attention trials have more variable lengths (based on resp time)
#read in behavioural data

df = pd.read_csv(beh_fname, sep = ',', header = 0)

if fid_sub in ['AttSacc_S01', 'AttSacc_S02']:
    nblocks, ntrials, nmeas = 24, 80, 2
else:
    nblocks, ntrials, nmeas = 12, 80, 2 #nmeas = number of things to extract from eye data, here just x and y

blockid = np.arange(1,nblocks+1)
blockid = np.repeat(blockid, ntrials)
df['block'] = blockid

sacctrl_df = df.query('task == 2')
saccblocks = pd.unique(sacctrl_df.block)
saccblockinds = np.subtract(saccblocks, 1)

timewin = [-300, 1000]
times = np.arange(timewin[0], timewin[1])
baseline     = np.divide([1920,1080], 2) # set x,y coords of where you want to baseline/correct the data to
baselinewindow = [int(np.where(times==0)[0]), int(np.where(times==0)[0])-150]

#all_epochs = np.zeros([nblocks, ntrials, nmeas, len(times)]) #premake epoched data structure
trig_to_find = '_ARR' #set string you want to find in the data, can be snippet if long msgs, or whole thing
sacc_epochs = np.zeros([nblocks*ntrials/2, nmeas, len(times)])

saccblock_epochs = np.zeros([len(saccblockinds), ntrials, nmeas, len(times)])
#%%
for i in range(len(saccblockinds)):
    blockind = i
    block = ds[saccblockinds[i]]
    events = copy.deepcopy(block['Msg'])
    evs    = np.array([i for event in events for i in event]) #flatten list of lists into list
    
    #remove uninformative messages (i.e. that get sent just after start of recording)
    uninf_inds = []
    for x,y in np.ndenumerate(evs):
        if y[2] == '!MODE':
            uninf_inds.append(x[0])
    evs = np.delete(evs, uninf_inds) #remove them. this now contains only task triggers
    
    #find triggers that you want
    trig_inds    = []
    for x,y in np.ndenumerate(evs):
        if trig_to_find in y[2]:
            trig_inds.append(x[0])
    #create mask to select only trigger of interest
    mask = np.zeros(len(evs), dtype = bool)
    mask[trig_inds] = True
    trigsoi = evs[mask] #get only triggers of interest
    
    #get trackertime start times for the trigger of interest
    epoch_starts = []
    for x,y in np.ndenumerate(trigsoi):
        epoch_starts.append(int(y[1]))
    
    
    starts = []
    for x,y in np.ndenumerate(epoch_starts):
        ind = int(np.where(block['trackertime'] == y)[0])
        starts.append(ind)
        
    times        = np.arange(timewin[0], timewin[1])
    trl_epochs   = np.zeros([ntrials, nmeas, len(times)])
    for x,y in np.ndenumerate(starts):
        try:
            epstart = y + timewin[0]
            epend   = y + timewin[1] 
            ep_lx   = block['lx'][epstart:epend]
            ep_ly   = block['ly'][epstart:epend] 
            ep_rx   = block['rx'][epstart:epend] 
            ep_ry   = block['ry'][epstart:epend]
            if len(ep_lx) < len(times): #too few samples, pad array with nans to the end of the time window
                #create empty vectors
                tmp_lx = np.full(len(times), np.NaN)
                tmp_ly = np.full(len(times), np.NaN)
                tmp_rx = np.full(len(times), np.NaN)
                tmp_ry = np.full(len(times), np.NaN)
                
                #fill first section with the data
                tmp_lx[:ep_lx.shape[0]] = ep_lx
                tmp_ly[:ep_lx.shape[0]] = ep_ly
                tmp_rx[:ep_lx.shape[0]] = ep_rx
                tmp_ry[:ep_lx.shape[0]] = ep_ry
                
                #now replace back into original structs
                ep_lx = tmp_lx
                ep_ly = tmp_ly
                ep_rx = tmp_rx
                ep_ry = tmp_ry
                
                
            #median baseline the data before averaging across eyes
            
            #get 100ms pre trigger median value
            lx_bl = np.nanmedian(ep_lx[baselinewindow[1]:baselinewindow[0]])
            ly_bl = np.nanmedian(ep_ly[baselinewindow[1]:baselinewindow[0]])
            rx_bl = np.nanmedian(ep_rx[baselinewindow[1]:baselinewindow[0]])
            ry_bl = np.nanmedian(ep_ry[baselinewindow[1]:baselinewindow[0]])
            
            #baseline epochs
            ep_lx = np.add(np.subtract(ep_lx, lx_bl),baseline[0])
            ep_ly = np.add(np.subtract(ep_ly, ly_bl),baseline[1])
            ep_rx = np.add(np.subtract(ep_rx, rx_bl),baseline[0])
            ep_ry = np.add(np.subtract(ep_ry, ry_bl),baseline[1])
            
            #average across eyes
            av_x = np.nanmean([ep_lx.T, ep_rx.T], axis = 0)
            av_y = np.nanmean([ep_ly.T, ep_ry.T], axis = 0)
            
            trl_epochs[x[0], 0, :] = av_x
            trl_epochs[x[0], 1, :] = av_y
            saccblock_epochs[blockind, x, 0, :] = av_x
            saccblock_epochs[blockind, x, 1, :] = av_y
        except ValueError:
                print x[0]
#%%
#saccblock_epochs contains x and y data for all saccade blocks, and all trials within that
#epoched relative to the appearance of target '_ARR' trigger

#check if epoch directories exist. if not, make them.
epcue_dir  = op.join(workingfolder, 'eyes','epoched_cue')
eptarg_dir = op.join(workingfolder, 'eyes','epoched_target')

if not op.exists(epcue_dir):
    os.mkdir(epcue_dir)
if not op.exists(eptarg_dir):
    os.mkdir(eptarg_dir)


#%%
tmp = copy.deepcopy(saccblock_epochs[0])
tmpx = np.vstack(np.squeeze(tmp[:,0,:]))

for i in range(tmpx.shape[0]):
    plt.figure()
    plt.axvline(0, ls = '--', color = '#636363')
    plt.plot(times, tmpx[i,:])
    plt.axhline(baseline[0], ls = '--', color = '#636363')     

#%%

#make block invariant
epochs_trls = np.vstack(all_epochs)


sacctrl_inds = [x for x in range(len(df)) if df.task[x] == 2]#

sacctrls = epochs_trls[sacctrl_inds]

#plot test
#tmp = np.squeeze(all_epochs[6,:,:,:]) #take first saccade block
#for x in range(sacctrls.shape[0]): #loop over all trials

plt.figure()
plt.axvline(0, ls = '--', color = '#636363')    
for x in range(960):
    plt.plot(times,sacc_epochs[x,0,:])


tmp_avx = np.nanmean(np.squeeze(tmp[:,0,:]), axis = 0)

plt.figure()
plt.plot(times, tmp_avx)
plt.axvline(0, ls='dashed', color = '#636363')








