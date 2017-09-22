# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:12:22 2017

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
import re
import os
import mne
import copy
import myfunctions
import cPickle
#from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
#%matplotlib
np.set_printoptions(suppress = True)
#%%
# set relevant directories and paths
#workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade/' #laptop directory
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
eyedat        = os.path.join(workingfolder, 'eyes')
behdat        = os.path.join(workingfolder, 'behaviour/csv')
eegdat        = os.path.join(workingfolder, 'EEG') #only on workstation though, no eeg data on laptop
eyelist       = os.listdir(eyedat)
eyelist       = np.sort(eyelist)
behlist       = np.sort(os.listdir(behdat))
saccadedat    = os.path.join(workingfolder, 'saccades')
if not os.path.exists(saccadedat):
    os.mkdir(saccadedat)
sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#sublist = [1, 2,    4, 5, 6, 7, 8, 9] #subject 3 has 961 trials in eyetracker data, and 960 in behavioural - check with nick. for now, remove from analysis
#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify


parts   = ['a','b']

#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify

epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
hrefsubs     = [7, 8]
lowsamp_subs = [4]

#1KHz  sample: 1 5 6 7 8 9
#250Hz sample: 4

# specs for the task
resxy = (1920,1080)
scrsize = (60,34)
scrdist = 100 # cm
#pixpcm = np.mean([resxy[0]/scrsize[0],resxy[1]/scrsize[1]])
#samplerate = 1000.0 # Hz , sample every 1ms. subject 4 has 250Hz SR, all other 1kHz
samplerate = 250.0 #for subject 4
# Data files
sep = '\t' #value separator
EDFSTART = "_BEG"
EDFSTOP  = "END"
TRIALORDER = ["B(.*)_T(.*)_BEG","B(.*)_T(.*)_CUE","B(.*)_T(.*)_ARR","B(.*)_T(.*)_RESP","B(.*)_T(.*)_END"]
INVALCODE = np.NaN
ppd = 72 #pixels per degree in current study

#%% combine eyetracking and behavioural data into a master list of dictionaries, and pickle into single file
pickname = os.path.join(workingfolder, 'preprocessed_eyes_9subs.pickle')
if not os.path.exists(pickname): #if the preprocessed data doesnt exist already, then run preprocessing and joining of behavioural and eye data
    edat = []
    for sub in range(len(sublist)):
        print('working on S%02d'%(sublist[sub]))

        if sub in range(0,2): #subjects 1 and 2 only have one file per subject
            fname    = os.path.join(eyedat, 'AttSacc_S%02d.asc'%(sublist[sub]));              datname  = os.path.join(behdat, 'AttSacc_S%02d.csv'%(sublist[sub]))
            bdata    = pd.DataFrame.from_csv(datname, header=0, sep = ',', index_col=False);  edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
        elif sub in range(2,9): #all other subjects have 2 files per sub
            fname    = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[0]));  datname  = os.path.join(behdat, 'AttSacc_S%02d%s.csv'%(sublist[sub], parts[0]))
            bdata    = pd.DataFrame.from_csv(datname, header=0, sep = ',', index_col=False);  edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
            if sublist[sub]==3: edata = edata[0:960]

            fname2   = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[1]));  datname2 = os.path.join(behdat, 'AttSacc_S%02d%s.csv'%(sublist[sub], parts[1]))
            bdata2   = pd.DataFrame.from_csv(datname2, header=0, sep = ',', index_col=False); edata2   = read_edf(fname2, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)

        if sublist[sub] in epsubs: #EP locations
            targlocations = np.array([[288,119], # cueloc = 1
                                     [119, 288], # cueloc = 2
                                     [-119, 288],# cueloc = 3
                                     [-288, 119],# cueloc = 4
                                     [-288,-119],# cueloc = 5
                                     [-119,-288],# cueloc = 6
                                     [119,-288], # cueloc = 7
                                     [288,-119]] # cueloc = 8
                                     )
        elif sublist[sub] in ohbasubs: #ohba locations
            targlocations = np.array([[399,165], # cueloc = 1
                                     [165, 399], # cueloc = 2
                                     [-165, 399],# cueloc = 3
                                     [-399, 165],# cueloc = 4
                                     [-399,-165],# cueloc = 5
                                     [-165,-399],# cueloc = 6
                                     [165,-399], # cueloc = 7
                                     [399,-165]] # cueloc = 8
                                     )

        targlocations[:,0] = targlocations[:,0]+(resxy[0]/2) # correct x from tracker coords to normal coords
        targlocations[:,1] = targlocations[:,1]+(resxy[1]/2) # correct y from tracker coords to normal coords
        
        if sub == 4: #this is subject 5. in trial 568, the time array is shifted by 1 sample, and skips sample 2762. need to realign ['time'] on this trial
            #length of trial 568 is 2936
            edata[568]['time'] = np.arange(0, len(edata[568]['x'])) #set the time to align to length of the trial

        if sub in range(0,2): #subjects 1 and 2 only have one file, not two so script changes accordingly.
            print('combining S%02d eyetracking and behavioural data, and adding triggers to eyetracking data'%(sublist[sub]))
            for trial in range(len(edata)):
                trl = bdata.iloc[trial,:]
                edata[trial]['behaviour'] = {
                    'subject'     : trl.loc['subject'] , 'session'  : trl.loc['session'] ,
                    'task'        : trl.loc['task']    , 'cuecol'   : trl.loc['cuecol']  ,
                    'cueloc'      : trl.loc['cueloc']  , 'validity' : trl.loc['validity'],
                    'targloc'     : trl.loc['targloc'] , 'targtilt' : trl.loc['targtilt'],
                    'delay'       : trl.loc['delay']   , 'resp'     : trl.loc['resp']    ,
                    'time'        : trl.loc['time']    , 'corr'     : trl.loc['corr']    ,
                    'sacc_allowed': 0,
                    'targlocpix'  : targlocations[int(trl.loc['targloc'])-1]
                }
                trigs = edata[trial]['events']['msg']
                if len(trigs) == 4: # attention trial
                    trltype = 1 #attention trial
                    begtrig   = edata[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                    resptrig  = edata[trial]['events']['msg'][3][0] #get edf timestamp for array trigger
                elif len(trigs) == 3: #saccade trial, no response
                    trltype = 2 #saccade trial
                    begtrig   = edata[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                    #find sample nearest to the trigger time
                begind  = np.argmin(np.abs(begtrig  - edata[trial]['trackertime']))
                cueind  = np.argmin(np.abs(cuetrig  - edata[trial]['trackertime']))
                arrind  = np.argmin(np.abs(arrtrig  - edata[trial]['trackertime']))
                if trltype == 1:
                    respind = np.argmin(np.abs(resptrig - edata[trial]['trackertime']))
                if trltype == 1: #attention trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],
                        "resp"  : [resptrig, respind]}
                if trltype == 2: #saccade trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],}
                edata[trial]['triggers'] = triggers
                # triggers : [EDFTIME, TRIAL_TIME]      
            saccfname = '%s/AttentionSaccade_S%02d_SaccadeData.csv' %(saccadedat, sublist[sub])
            if not os.path.exists(saccfname):    
                print('writing saccade data to file now')
                saccfile     = open(saccfname, 'w')
                saccfile.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                    'subject','trial','start','end','duration','startx', 'starty','endx', 'endy', 'velocity'))
                for trial in range(len(edata)):
                    saccades = edata[trial]['events']['Esac']
                    tnum = trial + 1
                    subject = edata[trial]['behaviour']['subject']
                    ttime = edata[trial]['trackertime'][0]
                    #saccades[saccade]: start, end, duration, startx, starty, endx, endy
                    for saccade in range(len(saccades)):
                        saccadevel = np.sqrt((saccades[saccade][5]-saccades[saccade][3])**2 +
                                             (saccades[saccade][6]-saccades[saccade][4])**2  ) / saccades[saccade][2]
                        saccfile.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                                subject,tnum,saccades[saccade][0]-ttime,saccades[saccade][1]-ttime,
                                saccades[saccade][2],saccades[saccade][3],saccades[saccade][4],
                                saccades[saccade][5],saccades[saccade][6],saccadevel))
                saccfile.close()
            comb = edata

        elif sub not in range(0,2): # subjects 3 onwards have two files, not one
            print('combining S%02d part 1 eyetracking and behavioural data, and adding triggers to eyetracking data'%(sublist[sub]))
            for trial in range(len(edata)):
                trl  = bdata.iloc[trial,:]
                edata[trial]['behaviour'] = {
                    'subject'    : trl.loc['subject'] , 'session'  : trl.loc['session'] ,
                    'task'       : trl.loc['task']    , 'cuecol'   : trl.loc['cuecol']  ,
                    'cueloc'     : trl.loc['cueloc']  , 'validity' : trl.loc['validity'],
                    'targloc'    : trl.loc['targloc'] , 'targtilt' : trl.loc['targtilt'],
                    'delay'      : trl.loc['delay']   , 'resp'     : trl.loc['resp']    ,
                    'time'       : trl.loc['time']    , 'corr'     : trl.loc['corr']    , 
                    'sacc_allowed': 0,
                    'targlocpix' : targlocations[int(trl.loc['targloc'])-1]
                    }
                trigs = edata[trial]['events']['msg']
                if len(trigs) == 4: # attention trial
                    trltype = 1 #attention trial
                    begtrig   = edata[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                    resptrig  = edata[trial]['events']['msg'][3][0] #get edf timestamp for array trigger
                elif len(trigs) == 3: #saccade trial, no response
                    trltype = 2 #saccade trial
                    begtrig   = edata[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                #find sample nearest to the trigger time
                begind  = np.argmin(np.abs(begtrig  - edata[trial]['trackertime']))
                cueind  = np.argmin(np.abs(cuetrig  - edata[trial]['trackertime']))
                arrind  = np.argmin(np.abs(arrtrig  - edata[trial]['trackertime']))
                if trltype == 1:
                    respind = np.argmin(np.abs(resptrig - edata[trial]['trackertime']))
                if trltype == 1: #attention trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],
                        "resp"  : [resptrig, respind]}
                if trltype == 2: #saccade trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],}
                edata[trial]['triggers'] = triggers
                # triggers : [EDFTIME, TRIAL_TIME]      
            saccfname = '%s/AttentionSaccade_S%02d%s_SaccadeData.csv' %(saccadedat, sublist[sub],parts[0])
            if not os.path.exists(saccfname):    
                print('writing saccade data to file now')
                saccfile     = open(saccfname, 'w')
                saccfile.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                    'subject','trial','start','end','duration','startx', 'starty','endx', 'endy', 'velocity'))
                for trial in range(len(edata)):
                    saccades = edata[trial]['events']['Esac']
                    tnum = trial + 1
                    subject = edata[trial]['behaviour']['subject']
                    ttime = edata[trial]['trackertime'][0]
                    #saccades[saccade]: start, end, duration, startx, starty, endx, endy
                    for saccade in range(len(saccades)):
                        saccadevel = np.sqrt((saccades[saccade][5]-saccades[saccade][3])**2 +
                                             (saccades[saccade][6]-saccades[saccade][4])**2  ) / saccades[saccade][2]
                        saccfile.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                                subject,tnum,saccades[saccade][0]-ttime,saccades[saccade][1]-ttime,
                                saccades[saccade][2],saccades[saccade][3],saccades[saccade][4],
                                saccades[saccade][5],saccades[saccade][6],saccadevel))
                saccfile.close()
            print('combining S%02d part 2 eyetracking and behavioural data, and adding triggers to eyetracking data'%(sublist[sub]))
            for trial in range(len(edata2)):
                trl          = bdata2.iloc[trial,:]
                edata2[trial]['behaviour'] = {
                    'subject'      : trl.loc['subject'] , 'session'  : trl.loc['session'] ,
                    'task'         : trl.loc['task']    , 'cuecol'   : trl.loc['cuecol']  ,
                    'cueloc'       : trl.loc['cueloc']  , 'validity' : trl.loc['validity'],
                    'targloc'      : trl.loc['targloc'] , 'targtilt' : trl.loc['targtilt'],
                    'delay'        : trl.loc['delay']   , 'resp'     : trl.loc['resp']    ,
                    'time'         : trl.loc['time']    , 'corr'     : trl.loc['corr']    ,
                    'targlocpix'   : targlocations[int(trl.loc['targloc'])-1],
                    'sacc_allowed' : 0
                }
                trigs = edata2[trial]['events']['msg']
                if len(trigs) == 4: # attention trial
                    trltype = 1 #attention trial
                    begtrig   = edata2[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata2[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata2[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                    resptrig  = edata2[trial]['events']['msg'][3][0] #get edf timestamp for array trigger
                elif len(trigs) == 3: #saccade trial, no response
                    trltype = 2 #saccade trial
                    begtrig   = edata2[trial]['events']['msg'][0][0] #get edf timestamp for the trial start trigger
                    cuetrig   = edata2[trial]['events']['msg'][1][0] #get edf timestamp for cue trigger
                    arrtrig   = edata2[trial]['events']['msg'][2][0] #get edf timestamp for array trigger
                #find sample nearest to the trigger time
                begind  = np.argmin(np.abs(begtrig  - edata2[trial]['trackertime']))
                cueind  = np.argmin(np.abs(cuetrig  - edata2[trial]['trackertime']))
                arrind  = np.argmin(np.abs(arrtrig  - edata2[trial]['trackertime']))
                if trltype == 1:
                    respind = np.argmin(np.abs(resptrig - edata2[trial]['trackertime']))
                if trltype == 1: #attention trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],
                        "resp"  : [resptrig, respind]}
                if trltype == 2: #saccade trial triggers
                    triggers = { #make dictionary of triggers
                        "start" : [begtrig,  begind],
                        "cue"   : [cuetrig,  cueind],
                        "array" : [arrtrig,  arrind],}
                edata2[trial]['triggers'] = triggers
                # triggers : [EDFTIME, TRIAL_TIME]  
            saccfname2    = '%s/AttentionSaccade_S%02d%s_SaccadeData.csv' %(saccadedat, sublist[sub],parts[1])
            if not os.path.exists(saccfname2):
                print('writing saccade data to file now')
                saccfile2     = open(saccfname2, 'w')
                saccfile2.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                    'subject','trial','start','end','duration','startx', 'starty','endx', 'endy', 'velocity'))
                for trial in range(len(edata2)):
                    saccades = edata2[trial]['events']['Esac']
                    tnum = trial + 1
                    subject = edata2[trial]['behaviour']['subject']
                    ttime = edata2[trial]['trackertime'][0]
                    #saccades[saccade]: start, end, duration, startx, starty, endx, endy
                    for saccade in range(len(saccades)):
                        saccadevel = np.sqrt((saccades[saccade][5]-saccades[saccade][3])**2 +
                                             (saccades[saccade][6]-saccades[saccade][4])**2  ) / saccades[saccade][2]
                        saccfile2.write('{},{},{},{},{},{},{},{},{},{} \n'.format(
                                subject,tnum,saccades[saccade][0]-ttime,saccades[saccade][1]-ttime,
                                saccades[saccade][2],saccades[saccade][3],saccades[saccade][4],
                                saccades[saccade][5],saccades[saccade][6],saccadevel))
                saccfile2.close()
            comb = edata + edata2
        edat.append(comb)

    edata = copy.deepcopy(edat) #so don't have to rerun preprocessing but instead can reload edat below
    print 'done'
else:
    print 'loading preprocessed data from pickle'
    with open(os.path.join(workingfolder, pickname), 'rb') as handle:
        edat = cPickle.load(handle)
        print 'done!'

print 'saving processed data in pickle format'
with open(os.path.join(workingfolder, pickname), 'w') as handle:
    cPickle.dump(edat, handle)
print 'done!'    
#%%
