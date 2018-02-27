#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:59:40 2018

@author: sammirc
"""



import numpy as np
import os.path as op
import os
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
#workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade' #laptop directory

os.chdir(workingfolder)
import BCEyes

np.set_printoptions(suppress = True)

eyedir        = os.path.join(workingfolder, 'eyes')
cleaned_dir   = os.path.join(eyedir, 'gaze_cleaned')

list_fnames= sorted(os.listdir(cleaned_dir))

#%%

fname = op.join(cleaned_dir, list_fnames[2])

print '\nloading parsed data from pickle'
with open(fname, 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'


nblocks, ntrials, nmeas = 12, 80, 2 #nmeas = number of things to extract from eye data, here just x and y

timewin = [-200, 500]
times = np.arange(timewin[0], timewin[1])

all_epochs = np.empty([nblocks, ntrials, nmeas, len(times)]) #premake epoched data structure




count = 0
for block in ds:
    events = copy.deepcopy(block['Msg'])
    evs    = np.array([i for event in events for i in event]) #flatten list of lists into list
    
    #remove uninformative messages (i.e. that get sent just after start of recording)
    uninf_inds = []
    for x,y in np.ndenumerate(evs):
        if y[2] == '!MODE':
            uninf_inds.append(x[0])
    evs = np.delete(evs, uninf_inds) #remove them. this now contains only task triggers
    
    #find triggers that you want
    trig_to_find = 'CUE' #set string you want to find in the data, can be snippet if long msgs, or whole thing
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
        
    timewin      = [-200, 500] #set time window for epoching
    times        = np.arange(timewin[0], timewin[1])
    epoched_data = np.empty([len(starts), 2, len(times)]) #preload array of appropriate size
    baseline     = np.divide([1920,1080], 2) # set x,y coords of where you want to baseline/correct the data to
    for x,y in np.ndenumerate(starts):
        epstart = y + timewin[0]
        epend   = y + timewin[1] #+1 to account for clipping at end of indexing
        ep_lx   = block['lx'][epstart:epend]
        ep_ly   = block['ly'][epstart:epend] 
        ep_rx   = block['rx'][epstart:epend] 
        ep_ry   = block['ry'][epstart:epend]
        #median baseline the data before averaging across eyes
        
        #get 100ms pre trigger median value
        lx_bl = np.nanmedian(ep_lx[100:201])
        ly_bl = np.nanmedian(ep_ly[100:201])
        rx_bl = np.nanmedian(ep_rx[100:201])
        ry_bl = np.nanmedian(ep_ry[100:201])
        
        #baseline epochs
        ep_lx = np.add(np.subtract(ep_lx, lx_bl),baseline[0])
        ep_ly = np.add(np.subtract(ep_ly, ly_bl),baseline[1])
        ep_rx = np.add(np.subtract(ep_rx, rx_bl),baseline[0])
        ep_ry = np.add(np.subtract(ep_ry, ry_bl),baseline[1])
        
        #average across eyes
        av_x = np.nanmean([ep_lx.T, ep_rx.T], axis = 0)
        av_y = np.nanmean([ep_ly.T, ep_ry.T], axis = 0)
        
        #store average of the two eyes
        all_epochs[count,x, 0, :] = av_x
        all_epochs[count,x, 1, :] = av_y
    count += 1
    





