# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:52:28 2017

@author: sammirc
"""


import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import pygazeanalyser
import matplotlib
from matplotlib import pyplot as plt
from pygazeanalyser.edfreader import read_edf
from pygazeanalyser import gazeplotter
from scipy import stats, signal, ndimage, interpolate
from scipy.signal import resample
import re
import os
import mne
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)
import myfunctions

#from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
%matplotlib
np.set_printoptions(suppress = True)

import multiprocessing
n_cores = multiprocessing.cpu_count() #8 cores on workstation


#%% load in all data from pickle (this is combined behavioural and eyetracking data, with x and y interpolated for blink reduction)

pickname = os.path.join(workingfolder, 'AttSacc_9subs_cleanedxy_BehavEyeDat.pickle')
print 'loading combined data from pickle'
with open(os.path.join(workingfolder, pickname), 'rb') as handle:
    edat = cPickle.load(handle)
print 'done!'

sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#%%

# get saccade time for saccade task trials
# finds the closest saccade of a tral to the array trigger (when target appears)
# takes time at which it starts, and subtracts from array onset to get saccade time

# also stores 'csac' which is the saccade closest to the target onset.
# If this is the first saccade after target (i.e appropriate response) then ['csac']==['fsac'] would be true,

# if using np.argmin then negative values mean saccades happened before target appeared!
#saccade response: start, end, duration, startx, starty, endx, endy

for sub in range(len(edat)):
    sacctask = [x for x in range(len(edat[sub])) if edat[sub][x]['behaviour']['task'] == 2.0]
    for i in sacctask:
        targettrig  = edat[sub][i]['triggers']['array'] #0 is the trackertime, 1 is the within-trial time
        fstime      = edat[sub][i]['trackertime'][0]
        trlsaccades = edat[sub][i]['events']['Esac']
        if len(trlsaccades) == 0:
            s_starts = np.array([]) #initialise empty array such that s_starts.size == 0 if no saccades in the trial
            pp_sac   = np.array([])
            c_sac    = np.array([])
        elif len(trlsaccades) == 1: #one saccade in the trial only
            s_starts    = np.array(edat[sub][i]['events']['Esac'])[0][0]   # start point of all saccades on this trial
            pp_sac      = np.squeeze(np.where(s_starts >= targettrig[0])) # all post prob saccades for this trial
            c_sac       = np.argmin(np.abs(s_starts - targettrig[0]))     # closest saccade to the target onset
        else:
            s_starts    = np.array(edat[sub][i]['events']['Esac'])[:,0]   # start point of all saccades on this trial
            pp_sac      = np.squeeze(np.where(s_starts >= targettrig[0])) # all post prob saccades for this trial
            c_sac       = np.argmin(np.abs(s_starts - targettrig[0]))     # closest saccade to the target onset
        if s_starts.size == 0: # no saccades at all in this trial
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = np.NaN # no first saccade as no saccades on trial
            edat[sub][i]['behaviour']['sacctime'] = np.NaN # ST is missing as no saccade on the trial
            edat[sub][i]['behaviour']['csac']     = np.NaN # no closest saccade as none on the trial
        elif s_starts.size > 0 and pp_sac.size == 0: #there were saccades on this trial, but none after the probe
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = np.NaN
            edat[sub][i]['behaviour']['sacctime'] = np.NaN
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]
        elif s_starts.size > 0 and pp_sac.size == 1: #saccades on this trial, and only one happened after the target appeared
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = trlsaccades[pp_sac] # first saccade after target onset
            edat[sub][i]['behaviour']['sacctime'] = trlsaccades[pp_sac][0] - targettrig[0] #diff in tracker time between saccade onset and target onset
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]     # saccade closest to the target onset
        elif s_starts.size > 0 and pp_sac.size > 1:
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = trlsaccades[pp_sac[0]] # first saccade after target onset
            edat[sub][i]['behaviour']['sacctime'] = trlsaccades[pp_sac[0]][0] - targettrig[0] #diff in tracker time between saccade onset and target onset
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]     # saccade closest to the target onset
#%%
#pipeline:
#   1. interpolate effect of blinks on the x and y traces (this is done in the preprocessing)
#   2. baseline x and y to fixation x (960) and y (540) prior to cue period (i.e. take average of 100ms before, remove it, then add that value to recentre traces)
#   3. euclidian distance measurements
          
#%% step 2 - baselining x and y 
          
fixpos = [960, 540] #centre of the screen

for sub in range(len(edat)):
    print('baselining x and y data for subject %s' %sublist[sub])
    for trl in range(len(edat[sub])):
        cueind = edat[sub][trl]['triggers']['cue'][1] #0 is in trackertime, 1 is in trial time 
        xbline = np.nanmean(edat[sub][trl]['x'][cueind-100:cueind])
        ybline = np.nanmean(edat[sub][trl]['y'][cueind-100:cueind])
        edat[sub][trl]['x'] = np.subtract(edat[sub][trl]['x'], (xbline-fixpos[0]))
        edat[sub][trl]['y'] = np.subtract(edat[sub][trl]['y'], (ybline-fixpos[1]))

#%% step 3 - distance measurements from fixation and target

# for each trial, calculate time series for distance from fixation (dist2fix) and distance from target (dist2fix)
# this collapses across x and y into one series. - will make baselining trials easy for both dist2targ and dist2fix as can just reduce one instead of 2
# but this may be inaccurate because may need to do this before doing this distance measurement, so it fixes both rather than just pixel distance. unsure?


#this is a bit slow, and could ultimately be multi-cored at a later point.
fixpos = [960,540] #centre of the screen
for sub in range(len(edat)):
    print('working on subject %s'%sublist[sub])
    for trl in range(len(edat[sub])):
        targpos   = np.squeeze(edat[sub][trl]['behaviour']['targlocpix']).tolist()
        eyepos    = np.vstack([edat[sub][trl]['x'], edat[sub][trl]['y']]) #stack x and y into array of shape (2, length of trial in samples)
        dist2fix  = np.zeros(eyepos.shape[1])
        dist2targ = np.zeros(eyepos.shape[1])
        for i in range(eyepos.shape[1]):
            idist = myfunctions.Eucdist( fixpos[0],  fixpos[1], eyepos[0,i], eyepos[1,i])
            itarg = myfunctions.Eucdist(targpos[0], targpos[1], eyepos[0,i], eyepos[1,i])
            #dist2fix.append(idist)
            #dist2targ.append(itarg)
            dist2fix[i] = idist
            dist2targ[i] = itarg
        edat[sub][trl]['dist2fix']  = dist2fix
        edat[sub][trl]['dist2targ'] = dist2targ
print('finished working on distance metrics')

#plt.figure();
#plt.plot(dist2fix)
#plt.axvline(dummy[0]['triggers']['array'][1], ls = 'dashed', color = '#636363');
#plt.axvline(dummy[0]['triggers']['cue'][1], ls = 'dashed', color = '#fc9272');
#plt.axvline(dummy[0]['behaviour']['fsac'][0]-dummy[0]['behaviour']['fstime'], ls = 'dashed', color = '#3182bd')
#
#plt.figure();
#plt.plot(dist2targ);
#plt.axvline(dummy[0]['triggers']['array'][1], ls = 'dashed', color = '#636363');
#plt.axvline(dummy[0]['triggers']['cue'][1], ls = 'dashed', color = '#fc9272');
#plt.axvline(dummy[0]['behaviour']['fsac'][0]-dummy[0]['behaviour']['fstime'], ls = 'dashed', color = '#3182bd')
#%% try a new way of detecting saccade closest to the target appearance (ignore the code section at the top)
            
for sub in range(len(edat)):
    sacctask = [x for x in range(len(edat))
                if edat[sub][x]['behaviour']['task'] == 2.0]:
    for sactrl in sacctask:
        #get some trial info needed for this
        dist2targ = edat[sub][sactrl]['dist2targ']            # trial distance from target
        arrind    = edat[sub][sactrl]['triggers']['array'][1] # onset of the target in trial time
        
        #calculate some things needed for this saccade detection
        gazevel  = np.abs(np.diff(dist2targ))                 # velocity of gaze within the trial
        baseline = np.nanmedian(gazevel[arrind-100:arrind])   # median velocity prior to target appearing (subs fixated at this point)
        
        posttarginds = [x for x in gazevel            # for all points in the trial
                        if gazevel[x] >= baseline*10  # if above velocity threshold (10 times baseline median)
                        and x >= arrind]              # and after the appearance of the target
        inds         = [x for x in gazevel
                        if gazevel[x] >=*10]



#%%
edata = copy.deepcopy(edat) # use edata instead so dont have to re-run scripts. help with debugging

#%%

for sub in range(len(edata)):
    print('working on subject %s' %sublist[sub])
    sacctask = [x for x in range(len(edata[sub])) if edata[sub][x]['behaviour']['task'] == 2.0]
    for trl in sacctask:
        #get some trial info
        dist2fix  = edata[sub][trl]['dist2fix']
        dist2targ = edata[sub][trl]['dist2targ']
        fsacind   = edata[sub][trl]['behaviour']['fsac'][0] - edata[sub][trl]['behaviour']['fstime'] #onset of first saccade in trial time
        targpos   = edata[sub][trl]['behaviour']['targlocpix'] #location of target on screen in pixels
        
        #calculations needed for this section
        gazevel  = np.abs(np.diff(dist2targ)) #velocity of gaze trace (rel 2 target location)
        baseline = np.nanmedian(gazevel[fsacind-150:fsacind]) #pre-saccade baseline velocity for threshold setting
        
        velinds = [x for x
                   in range(len(gazevel))
                   if gazevel[x] >= 10*baseline         # sample must exceed the baseline velocity x 10 to classify as saccadic velocity
                   and x in range(fsacind,fsacind+200)] # and must be within 100ms of saccade onset
        saccstart, saccend = velinds[0], velinds[-1]    # take the first and last parts (i.e the tips of the rising edge of the saccade)
        fixed = np.nanmedian(dist2targ[saccend:saccend+100])
        x, y = np.nanmedian(edat[sub][trl]['x'][saccend:saccend+100]), np.nanmedian(edat[sub][trl]['y'][saccend:saccend+100])
        
        # save some values back into the data structure for later analysis
        # fixed euclidian distance (saccade response accuracy)
        # pixel location of saccade response (x,y)
        # distances between target and saccade response pos + between target and fixation
        # dist2targ, distfromfix and targfromfix form triangle for calculating angular error in saccade response (another measure of accuracy)
        
        sacresponse = { #store values relating to the saccade response
        'dist2targ'   : fixed,
        'resp_pix'    : (x,y),
        'distfromfix' : myfunctions.Eucdist(targpos[0], targpos[1], x, y), #straight line distance between fixation and saccade pos
        'targfromfix' : myfunctions.Eucdist(fixpos[0], fixpos[1], targpos[0], targpos[1]) #straight line dist between target and fixation locs
        }
        
        edata[sub][trl]['saccade_response'] = sacresponse
#%%
sub = copy.deepcopy(edata[2])
sacctask = [x for x in range(len(sub)) if sub[x]['behaviour']['task'] == 2.0]
#%%
for trial in sacctask:
    #get some trial info for this step
    dist2targ = sub[trial]['dist2targ']
    dist2fix  = sub[trial]['dist2fix']
    targpos   = sub[trial]['behaviour']['targlocpix'].tolist()
    
    arrind    = sub[trial]['triggers']['array'][1] #trial-time of target onset
    cueind    = sub[trial]['triggers']['cue'][1]
    
    #calculations needed for this part
    gazevel_abs  = np.abs(np.diff(dist2targ))
    gazevel      = np.diff(dist2targ)
    baseline_cue = np.nanmedian(dist2targ[cueind-100:cueind]) #baseline to pre-cue (this may be heavily contaminated by blinks and/or saccades back to fixation)
    baseline_arr = np.nanmedian(dist2targ[arrind-100:arrind]) # most likely noise source at this point is blinks
     
#%%
#plt.close('all')
trial = -12

velthresh = 25
sub[sacctask[trial]]['behaviour']['validity']

dist2targ = sub[sacctask[trial]]['dist2targ']
dist2fix  = sub[sacctask[trial]]['dist2fix']
targpos   = sub[sacctask[trial]]['behaviour']['targlocpix'].tolist()

arrind    = sub[sacctask[trial]]['triggers']['array'][1] #trial-time of target onset
cueind    = sub[sacctask[trial]]['triggers']['cue'][1]

#calculations needed for this part
gazevel_abs  = np.abs(np.diff(dist2targ))
gazevel      = np.diff(dist2targ)

baseline_cue_abs = np.nanmedian(gazevel_abs[cueind-100:cueind]) #baseline to pre-cue (this may be heavily contaminated by blinks and/or saccades back to fixation)
baseline_arr_abs = np.nanmedian(gazevel_abs[arrind-100:arrind]) # most likely noise source at this point is blinks

baseline_cue = np.nanmedian(gazevel[cueind-100:cueind]) #baseline to pre-cue (this may be heavily contaminated by blinks and/or saccades back to fixation)
baseline_arr = np.nanmedian(gazevel[arrind-100:arrind])

start_thresh = -baseline_arr_abs*velthresh
end_thresh   = baseline_arr_abs*velthresh


# get points where the velocity is lower than the threshold
# reflecting a quick drop to zero indicative of the start of a saccade
vel_start = [x for x in range(len(gazevel))
             if gazevel[x] <= start_thresh
             and x > arrind]

# high positive velocities indicate a move away from 0 (here, this is distance from target, so means getting away from it)
vel_end   = [x for x in range(len(gazevel))
             if gazevel[x] >= end_thresh
             and x > arrind]
# np.where(vel_start <= arrind+400) ## this will get starts within 400ms of the array (i.e valid saccades!)


#plot gaze velocities, as the absolute and non-absolute might have important different uses
plt.figure()
plt.plot(gazevel, color = '#a6cee3', label = 'gaze velocity')
plt.axhline(baseline_cue*velthresh, color = '#d6604d', ls = 'dashed', label = '15 x pre-cue baseline')
plt.axhline(baseline_arr*velthresh, color = '#b2182b', ls = 'dashed', label = '15 x pre-target baseline')
plt.axhline(start_thresh, color = '#542788', ls = 'dashed', label = '15 x abs pre-target baseline')
plt.axhline(end_thresh, color = '#b35806', ls = 'dashed', label = '15 x abs pre-target baseline')
plt.axvline(arrind, ls = 'dashed')
plt.axvline(vel_start[0], ls = 'dashed', color = '#35978f')
plt.axvline(vel_end[0], ls = 'dashed', color = '#35978f')
plt.axvline(inds[np.where(inds>arrind)][0], ls = 'dashed', color = '#636363')
plt.axvline(inds[np.where(inds>arrind)][1], ls = 'dashed', color = '#636363')

plt.xlim([arrind-200, None])
plt.title('trial %s'%sacctask[trial])
plt.legend(loc = 2)



plt.figure()
plt.plot(dist2targ, label = 'distance from target location', color = '#1a1a1a')
plt.axvline(arrind, ls = 'dashed')
plt.axvline(vel_start[0], ls = 'dashed', color = '#35978f')
plt.axvline(vel_end[0], ls = 'dashed', color = '#35978f')
plt.axvline(inds[np.where(inds>arrind)][0], ls = 'dashed', color = '#636363')
plt.xlim([arrind-200, None])
plt.title('trial %s'%sacctask[trial])
plt.legend(loc = 2)

inds_abs = peakutils.indexes(gazevel_abs, thres=0.7, min_dist = 15)
sacs_abs = inds_abs[np.where(inds_abs>arrind)].tolist()

#%%
plt.close('all')
# here it gets two inds -- the big positive velocities indicating the small spike after initial saccade, and the return to fixation
# this is great for positive peaks. need the negative peaks from gazevel for saccade onset
inds     = peakutils.indexes(gazevel, thres = 0.7, min_dist = 10) 
sacs     = inds[np.where(inds>arrind)].tolist()

plt.figure()
plt.title('velocity profile with peaks from peakutils indexes')
plt.plot(gazevel, color = '#756bb1', label = 'gaze velocity')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
for i in sacs:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-100, None])
plt.legend(loc=2)


plt.figure()
plt.plot(dist2targ, label = 'distance from target location', color = '#1a1a1a')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
for i in sacs:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-200, None])
plt.title('trial %s'%sacctask[trial])
plt.legend(loc = 2)

#%%

inds_neg = peakutils.indexes(gazevel_abs, thres = 0.9, min_dist = 10)
sacs_neg = inds_neg[np.where(inds_neg>arrind)].tolist()

plt.figure()
plt.title('velocity profile with peaks from peakutils indexes')
plt.plot(gazevel, color = '#756bb1', label = 'gaze velocity')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
for i in sacs_neg:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-100, None])
plt.legend(loc=2)

plt.figure()
plt.title('absolute velocity profile with peaks from peakutils indexes')
plt.plot(gazevel_abs, color = '#756bb1', label = 'absolute gaze velocity')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
for i in sacs_neg:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-100, None])
plt.legend(loc=2)


plt.figure()
plt.plot(dist2targ, label = 'distance from target location', color = '#1a1a1a')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
for i in sacs_neg:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-200, None])
plt.title('trial %s'%sacctask[trial])
plt.legend(loc = 2)


#%%
trial = -12

sub[sacctask[trial]]['behaviour']['validity']

dist2targ = sub[sacctask[trial]]['dist2targ']
dist2fix  = sub[sacctask[trial]]['dist2fix']
targpos   = sub[sacctask[trial]]['behaviour']['targlocpix'].tolist()

arrind    = sub[sacctask[trial]]['triggers']['array'][1] #trial-time of target onset
cueind    = sub[sacctask[trial]]['triggers']['cue'][1]

# find peaks in the velocity profile of dist2targ
peaks = 

#%% here is a way of doing it that may work, to find the negative deflection
trial = -12
trial = -28

sub[sacctask[trial]]['behaviour']['validity'] #valid or invalid trial
#get some trial info
dist2targ   = sub[sacctask[trial]]['dist2targ']
arrind      = sub[sacctask[trial]]['triggers']['array'][1]

gazevel_abs = np.abs(np.diff(dist2targ))
gazevel     = np.diff(dist2targ)

baseline    = np.nanmedian(gazevel_abs[arrind-100:arrind])

velrange_neg    = min(gazevel) - baseline
neg_thresh      = 0.8*velrange_neg 

velrange_pos    = max(gazevel) - baseline
pos_thresh      = 0.8*velrange_pos

peaks_neg = [x for x in range(len(gazevel)) if gazevel[x] < neg_thresh]
peaks_pos = [x for x in range(len(gazevel)) if gazevel[x] > pos_thresh]

sac_on    = peaks_neg[0] 





plt.figure()
plt.plot(dist2targ)
plt.axvline(arrind, ls = 'dashed', color = '#636363')
plt.axhline(np.nanmedian(dist2targ[peaks_neg[0]+30:peaks_neg[0]+130]), ls = 'dashed', color = '#31a354', label = 'post-onset nan median')
plt.axvline(peaks_neg[0]+30 , ls = 'dashed', color = '#addd8e')
plt.axvline(peaks_neg[0]+130, ls = 'dashed', color = '#addd8e')
plt.axhline(np.nanmedian(dist2targ[peaks_neg[0]+15:peaks_pos[0]-15 ]) , ls = 'dashed', color = '#d95f0e', label = 'between-peak nan median')
plt.axvline(peaks_neg[0]+15 , ls = 'dashed', color = '#fec44f')
plt.axvline(peaks_pos[0]-15, ls = 'dashed', color = '#fec44f')
plt.xlim([arrind-100,None])
plt.title('trial %s'%trial)
plt.legend(loc = 4)


plt.figure()
plt.title('trial %s'%trial)
plt.plot(gazevel, color = '#756bb1', label = 'gaze velocity')
plt.axvline(arrind, color = '#3182bd', ls = 'dashed', label = 'target onset')
plt.axvline(peaks_neg[0]+30 , ls = 'dashed', color = '#addd8e')
for i in peaks_pos:
    plt.axvline(i, ls = 'dashed', color = '#bdbdbd')
plt.xlim([arrind-100, None])
plt.legend(loc=2)


