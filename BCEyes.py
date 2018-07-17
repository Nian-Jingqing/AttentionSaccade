# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a series of functions designed to aid the analysis of eyetracking data
collected in the Brain and Cognition Lab.

to use this, import the script when you are loading in other packages in your script, e.g.:

import BCEyes

Author: Sammi Chekroud

"""
import numpy   as np
import scipy   as sp
from scipy import interpolate
import copy
import sys
import os


class DataError(Exception):
    '''
    Raised when there is an error in the datafile and expectations do not align with what is present
    '''
    def _init_(self, msg):
        self.msg = msg

class ArgumentError(Exception):
    """
    Raised when there is an error in the arguments used in a function here
    """
    def _init_(self, msg):
        self.msg = msg

def Eucdist(x1, y1, x2, y2):
    """
    calculate euclidian distance between two points

    formula: sqrt( (x2-x1)^2 + (y2-y1)^2 )

    """
    distance = np.sqrt( (x2-x1)**2 + (y2-y1)**2)
    return distance

def parse_eye_data(eye_fname, block_rec, trial_rec, nblocks, ntrials = None, binocular = True):
    """

    eye_fname    -- full path to a file to be parsed. this is used as a template to create the parsed data in pickle form

    block_rec    -- boolean as to whether you stopped/started recording blockwise in your task

    trial_rec    -- boolean as to whethre you stopped/started recording trialwise in your task

    nblocks      -- the number of blocks of task run in your experiment. This must be defined in all cases.

    ntrials      -- the number of trials in your data if you recorded the eyes with trialwise stop/start
                    if you recorded blockwise, then ntrials is not needed.

                    if you recorded trialwise, then the number of trials per block
                    is calculated and used to separate the data into defined blocks
    binocular    -- this is a boolean specifying if you made a binocular recording (True if yes, False if monocular)
    """
    if not os.path.exists(eye_fname): #check that the path to the file exists
        raise Exception('the filename: %s does not exist. Please check!' %eye_fname)

    if 'block_rec' not in locals() and 'trial_rec' not in locals():
        raise ArgumentError('Please indicate whether the recording was stopped/started on each trial or each block')

    if 'block_rec' in locals():
        if block_rec == True:
            trial_rec = False
    if 'trial_rec' in locals():
        if trial_rec == True:
            block_rec = False


    if 'block_rec' in locals():
        if block_rec == True and 'nblocks' not in locals():
            raise ArgumentError('Please specify how many blocks you expect in your data recording')

    if 'trial_rec' in locals():
        if trial_rec == True and 'ntrials' not in locals():
            raise ArgumentError('Please specify how many trials you expect in your data recording')
        elif trial_rec == True and 'ntrials' in locals() and ntrials == None:
            raise ArgumentError('Please specify a number of trials to expect')
        elif trial_rec == True and 'ntrials' in locals() and ntrials != None and 'nblocks' not in locals():
            raise ArgumentError('Please specify how many blocks of data you recorded')


    if block_rec:
        d = _parse_eye_data_blockwise(eye_fname, nblocks, binocular)
    else:
        d = _parse_eye_data_trialwise(eye_fname, nblocks, ntrials, binocular)

    return d #return the parsed data


def _parse_eye_data_blockwise(eye_fname, nblocks, binocular):
    """
    this function is called by parse_eye_data and will operate on data where
    the recording was stopped/started for each block of task data
    """

    d = open(eye_fname, 'r')
    raw_d = d.readlines()
    d.close()

    split_d = []
    for i in range(len(raw_d)):
        tmp = raw_d[i].split()
        split_d.append(tmp)

    start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 6 and split_d[x][0] == 'START']
    if len(start_inds) != nblocks:
        raise DataError('%d blocks are found in the data, not %d as has been input in nblocks' %(len(start_inds),nblocks))

    #get points where recording stopped
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']

    if len(start_inds) != len(end_inds):
        raise DataError('the number of times the recording was started and stopped does not align. check problems with acquisition')

    #assign some empty lists to get filled with information
    if binocular == True:
        traces = ['lx', 'rx', 'ly', 'ry', 'lp', 'rp']
    elif binocular == False:
        traces = ['x', 'y', 'p']
    
    blocked_data = np.array([]);
    

    for istart in range(len(start_inds)):

        tmpdata = dict() #this will house the arrays that we fill with information
        tmpdata['trackertime'] = np.array([])
        for trace in traces:
            tmpdata[trace] = np.array([])
            
        tmpdata['Efix'] = np.array([]); tmpdata['Sfix'] = np.array([])
        tmpdata['Esac'] = np.array([]); tmpdata['Ssac'] = np.array([])
        tmpdata['Eblk'] = np.array([]); tmpdata['Sblk'] = np.array([])
        tmpdata['Msg']  = np.array([])

        start_line = start_inds[istart]
        end_line   = end_inds[istart]

        iblk = np.array(split_d[start_line:end_line]) #get only lines with info for this block

        iblk_event_inds, iblk_events     = [], []
        iblk_blink_inds, iblk_blink      = [], []
        iblk_fix_inds, iblk_fix          = [], []
        iblk_sac_inds, iblk_sac          = [], []
        iblk_input_inds                  = []
        for x,y in np.ndenumerate(iblk):
            if y[0] == 'MSG':
                iblk_event_inds.append(x[0]) # add line index where a message was sent to the eyetracker
                iblk_events.append(y)        # add the line itself to list
            elif y[0] in ['EFIX', 'SFIX']:
                iblk_fix_inds.append(x[0])   # add line index where fixation detected (SR research)
                iblk_fix.append(y)           # add fixation event structure to list
            elif y[0] in ['ESACC', 'SSACC']:
                iblk_sac_inds.append(x[0])   # add line index where saccade detected (SR research)
                iblk_sac.append(y)           # add saccade event structure to list
            elif y[0] in ['EBLINK', 'SBLINK']:
                iblk_blink_inds.append(x[0]) # add line index where blink detected (SR research)
                iblk_blink.append(y)         # add blink event structure to list
            elif y[0] == 'INPUT':
                iblk_input_inds.append(x[0])  # find where 'INPUT' is in data (sometimes appears, has no use...)
               
        #the block events should really be an M x 3 shape array (because ['MSG', timestamp, trigger]).
        # if this isnt the case, you can't coerce to a shaped array (will be an array of lists :( ))
        #so find where this fails, and remove that event (it's likely to be: ['MSG', time_stamp, '!MODE', 'RECORD' ...]) as this is another silly line from eyelink
        
        events_to_remove = [x for x in range(len(iblk_events)) if len(iblk_events[x]) != 3]
        if len(events_to_remove) > 1:
            print ('warning, there are multiple trigger lines that have more than 3 elements to the line, check the data?')
        iblk_events.pop(events_to_remove[0]) #remove the first instance of more than 3 elements to the trigger line. should now be able to coerce to array with shape Mx3
        iblk_events = np.array(iblk_events)  #coerce to array for easier manipulation later on
        
        
        #get all non-data line indices
        iblk_nondata    = sorted(iblk_blink_inds + iblk_sac_inds + iblk_fix_inds + iblk_event_inds + iblk_input_inds)

        iblk_data = np.delete(iblk, iblk_nondata) #remove these lines so all you have is the raw data

        iblk_data = iblk_data[6:] #remove first five lines as these are filler after the recording starts

        try:
            iblk_data = np.vstack(iblk_data)
        except ValueError:
            for item in range(len(iblk_data)):
                iblk_data[item] = iblk_data[item][:7] #make sure each item has only 7 strings. sometimes the eyetracker hasn't appended the random .... at the end
            iblk_data = np.vstack(iblk_data) #shape of this should be number of columns in the file (data)


        iblk_data = iblk_data[:,:7]            # only take first 7 columns as these contain data of interest. last one is redundant (if exists)

        #before you can convert to float, need to replace missing data where its '.' as nans (in pupil this is '0.0')
        eyecols = [1,2,4,5] #leftx, lefty, rightx, right y col indices
        for col in eyecols:
            missing_inds = np.where(iblk_data[:,col] == '.') #find where data is missing in the gaze position, as probably a blink (or its missing as lost the eye)
            for i in missing_inds:
                iblk_data[i,col] = np.NaN #replace missing data ('.') with NaN
                iblk_data[i,3]   = np.NaN #replace left pupil as NaN (as in a blink)
                iblk_data[i,6]   = np.NaN #replace right pupil as NaN (as in a blink)

        iblk_data = iblk_data.astype(np.float) # convert data from string to floats for computations

        #for binocular data, the shape is:
        # columns: time stamp, left x, left y, left pupil, right x, right y, right pupil
        tmpiblkdata = dict()
        tmpiblkdata['iblk_trackertime'] = iblk_data[:,0]
        if binocular:
            tmpiblkdata['iblk_lx'] = iblk_data[:,1]
            tmpiblkdata['iblk_ly'] = iblk_data[:,2]
            tmpiblkdata['iblk_lp'] = iblk_data[:,3]
            tmpiblkdata['iblk_rx'] = iblk_data[:,4]
            tmpiblkdata['iblk_ry'] = iblk_data[:,5]
            tmpiblkdata['iblk_rp'] = iblk_data[:,6]
        elif not binocular:
            tmpiblkdata['iblk_x'] = iblk_data[:,1]
            tmpiblkdata['iblk_y'] = iblk_data[:,2]
            tmpiblkdata['iblk_p'] = iblk_data[:,3]


        # split Efix/Sfix and Esacc/Ssacc into separate lists, and make into arrays for easier manipulation later on
        iblk_efix = np.array([iblk_fix[x] for x in range(len(iblk_fix)) if
                     iblk_fix[x][0] == 'EFIX'])
        iblk_sfix = np.array([iblk_fix[x] for x in range(len(iblk_fix)) if
                     iblk_fix[x][0] == 'SFIX'])
        iblk_ssac = np.array([iblk_sac[x] for x in range(len(iblk_sac)) if
                     iblk_sac[x][0] == 'SSACC'])
        iblk_esac = np.array([iblk_sac[x] for x in range(len(iblk_sac)) if
                     iblk_sac[x][0] == 'ESACC'])
        iblk_sblk = np.array([iblk_blink[x] for x in range(len(iblk_blink)) if
                     iblk_blink[x][0] == 'SBLINK'])
        iblk_eblk = np.array([iblk_blink[x] for x in range(len(iblk_blink)) if
                     iblk_blink[x][0] == 'EBLINK'])

        #create tmpdata (the block structure) by adding in relevant information
        for trace in traces:
            tmpdata[trace] = tmpiblkdata['iblk_' + trace]
        
        tmpdata['trackertime'] = tmpiblkdata['iblk_trackertime']
        tmpdata['Efix'] = iblk_efix   #this should have 8  columns
        tmpdata['Sfix'] = iblk_sfix   #this should have 3  columns
        tmpdata['Esac'] = iblk_esac   #this should have 11 columns
        tmpdata['Ssac'] = iblk_ssac   #this should have 3  columns
        tmpdata['Eblk'] = iblk_eblk   #this should have 5  columns
        tmpdata['Sblk'] = iblk_sblk   #this should have 3  columns
        tmpdata['Msg']  = iblk_events #this should have 3  columns

        
        #tmpdata now contains the information for that blocks data. now we just need to add this to the blocked_data object before returning it
        
        blocked_data = np.append(blocked_data, copy.deepcopy(tmpdata))
    return blocked_data

def _parse_eye_data_trialwise(eye_fname, nblocks, ntrials):

    """
    this function is called by parse_eye_data and will operate on data where the recording was stopped/started for each trial of task data

    """

    d = open(eye_fname, 'r')
    raw_d = d.readlines()
    d.close()

    split_d = []
    for i in range(len(raw_d)):
        tmp = raw_d[i].split()
        split_d.append(tmp)

    #get all lines where 'START'i s seen, as this marks the start of the recording
    start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 6 and split_d[x][0] == 'START']
    if len(start_inds) != ntrials:
        raise DataError('%d trials are found in the data, not %d as has been input in ntrials' %(len(start_inds),ntrials))

    #get points where recording stopped
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']

    if len(start_inds) != len(end_inds):
        raise DataError('the number of times the recording was started and stopped does not align. check problems with acquisition')

    #assign some empty lists to get filled with information
    trackertime = []
    lx   = []; rx   = []
    ly   = []; ry   = []
    lp   = []; rp   = []


    Efix = []; Sfix = []
    Esac = []; Ssac = []
    Eblk = []; Sblk = []
    Msg  = []

    for i in range(len(start_inds)):
        start_line = start_inds[i]
        end_line   = end_inds[i]

        itrl = np.array(split_d[start_line:end_line])

        itrl_event_inds, itrl_events     = [], []
        itrl_blink_inds, itrl_blink      = [], []
        itrl_fix_inds, itrl_fix          = [], []
        itrl_sac_inds, itrl_sac          = [], []
        itrl_input_inds                  = []
        for x,y in np.ndenumerate(itrl):
            if y[0] == 'MSG':
                itrl_event_inds.append(x[0]) # add line index where a message was sent to the eyetracker
                itrl_events.append(y)        # add the line itself to list
            elif y[0] in ['EFIX', 'SFIX']:
                itrl_fix_inds.append(x[0])   # add line index where fixation detected (SR research)
                itrl_fix.append(y)           # add fixation event structure to list
            elif y[0] in ['ESACC', 'SSACC']:
                itrl_sac_inds.append(x[0])   # add line index where saccade detected (SR research)
                itrl_sac.append(y)           # add saccade event structure to list
            elif y[0] in ['EBLINK', 'SBLINK']:
                itrl_blink_inds.append(x[0]) # add line index where blink detected (SR research)
                itrl_blink.append(y)         # add blink event structure to list
            elif y[0] == 'INPUT':
               itrl_input_inds.append(x[0])  # find where 'INPUT' is in data (sometimes appears, has no use...)

        #get all non-data line indices
        itrl_nondata    = sorted(itrl_blink_inds + itrl_sac_inds + itrl_fix_inds + itrl_event_inds + itrl_input_inds)

        itrl_data = np.delete(itrl, itrl_nondata) #remove these lines so all you have is the raw data

        itrl_data = itrl_data[6:] #remove first five lines as these are filler after the recording starts

        try:
            itrl_data = np.vstack(itrl_data) #if this fails its most likely because the end string of dots is missing for some reason
        except ValueError: #if fails, fix
            for item in range(len(itrl_data)):
                itrl_data[item] = itrl_data[item][:7] #make sure each item has only 7 strings. sometimes the eyetracker hasn't appended the random .... at the end

        itrl_data = itrl_data[:,:7]            # only take first 6 columns as these contain data of interest (redundant if above section trips up

        itrl_data = np.vstack(itrl_data) #shape of this should be number of columns in the file (data)

        #before you can convert to float, need to replace missing data where its '.' as nans (in pupil this is '0.0')
        eyecols = [1,2,4,5] #leftx, lefty, rightx, right y col indices
        for col in eyecols:
            missing_inds = np.where(itrl_data[:,col] == '.') #find where data is missing in the gaze position, as probably a blink (or its missing as lost the eye)
            for i in missing_inds:
                itrl_data[i,col] = np.NaN #replace missing data ('.') with NaN
                itrl_data[i,3]   = np.NaN #replace left pupil as NaN (as in a blink)
                itrl_data[i,6]   = np.NaN #replace right pupil as NaN (as in a blink)
        itrl_data = itrl_data.astype(np.float) #convert data from string to floats for computations

        #for binocular data, the shape is:
        # columns: time stamp, left x, left y, left pupil, right x, right y, right pupil
        itrl_trackertime = itrl_data[:,0]
        itrl_lx, itrl_ly, itrl_lp = itrl_data[:,1], itrl_data[:,2], itrl_data[:,3]
        itrl_rx, itrl_ry, itrl_rp = itrl_data[:,4], itrl_data[:,5], itrl_data[:,6]

        # split Efix/Sfix and Esacc/Ssacc into separate lists
        itrl_efix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                     itrl_fix[x][0] == 'EFIX']
        itrl_sfix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                     itrl_fix[x][0] == 'SFIX']

        itrl_ssac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                     itrl_sac[x][0] == 'SSACC']
        itrl_esac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                     itrl_sac[x][0] == 'ESACC']

        itrl_sblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                     itrl_blink[x][0] == 'SBLINK']
        itrl_eblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                     itrl_blink[x][0] == 'EBLINK']

        #append to the collection of all data now
        trackertime.append(itrl_trackertime)
        lx.append(itrl_lx)
        ly.append(itrl_ly)
        rx.append(itrl_rx)
        ry.append(itrl_ry)
        lp.append(itrl_lp)
        rp.append(itrl_rp)
        Efix.append(itrl_efix)
        Sfix.append(itrl_sfix)
        Ssac.append(itrl_ssac)
        Esac.append(itrl_esac)
        Sblk.append(itrl_sblk)
        Eblk.append(itrl_eblk)
        Msg.append(itrl_events)

    trialsperblock = ntrials/nblocks
    rep_inds  = np.repeat(np.arange(nblocks),trialsperblock)
    #plt.hist(rep_inds, bins = nblocks) # all bars should be the same height now if its work? @ trials per block (80)

    iblock = {
    'trackertime': [],
    'lx'  : [], 'ly'  : [], 'lp'  : [],
    'rx'  : [], 'ry'  : [], 'rp'  : [],
    'Efix': [], 'Sfix': [],
    'Esac': [], 'Ssac': [],
    'Eblk': [], 'Sblk': [],
    'Msg' : []
    }
    dummy = copy.deepcopy(iblock)
    blocked_data = np.repeat(dummy, nblocks)

    for i in np.arange(nblocks):
        inds = np.squeeze(np.where(rep_inds == i))
        iblock_data = copy.deepcopy(iblock)
        for ind in inds: #add trialwise info into the blocked structure, continuous data rather than sectioned
            iblock_data['trackertime'].append(trackertime[ind])
            iblock_data['lx'].append(lx[ind])
            iblock_data['ly'].append(ly[ind])
            iblock_data['lp'].append(lp[ind])
            iblock_data['rx'].append(rx[ind])
            iblock_data['ry'].append(ry[ind])
            iblock_data['rp'].append(rp[ind])
            iblock_data['Efix'].append(Efix[ind])
            iblock_data['Sfix'].append(Sfix[ind])
            iblock_data['Esac'].append(Esac[ind])
            iblock_data['Ssac'].append(Ssac[ind])
            iblock_data['Eblk'].append(Eblk[ind])
            iblock_data['Sblk'].append(Sblk[ind])
            iblock_data['Msg'].append(Msg[ind])
            blocked_data[i] = iblock_data
    for block in range(len(blocked_data)): #concatenate trialwise signals into whole block traces to make artefact removal easier
        blocked_data[block]['trackertime'] = np.hstack(blocked_data[block]['trackertime'])
        blocked_data[block]['lx']          = np.hstack(blocked_data[block]['lx']         )
        blocked_data[block]['ly']          = np.hstack(blocked_data[block]['ly']         )
        blocked_data[block]['lp']          = np.hstack(blocked_data[block]['lp']         )
        blocked_data[block]['rx']          = np.hstack(blocked_data[block]['rx']         )
        blocked_data[block]['ry']          = np.hstack(blocked_data[block]['ry']         )
        blocked_data[block]['rp']          = np.hstack(blocked_data[block]['rp']         )
    return blocked_data




def find_missing_periods(data, nblocks, traces_to_scan):
    '''
    this function will find all sections of missing data. this typically relates to blinks
    but it will also find small patches where an eye has been dropped from the data.
    this just finds all patches, no matter the size, that are missing and outputs meaningful info into your data structure

    data argument expects a list, where each item in the list is a dictionary containing the data for a continuous block.

    binocular recorded data is expected here, where in each block the following keys are in the data dict:
        lx, ly, rx, ry.
    these correspond to the left and right gaze values. this function will iterate over each one to identify missing periods
    in each signal, and then output this information into new keys in your data:
    Eblk_lx, Eblk_ly, Eblk_rx, Eblk_ry


    traces_to_scan -- a list of the traces that you actually want to scan for blinks is needed here
                      this allows some flexibility in what you scan (e.g. if not at all interested in gaze, you can just find missing periods for the pupil data)


    example:
        data = find_missing_periods(data, nblocks = 1, traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])


    '''

    if not isinstance(data, np.ndarray):
        raise Exception('check the format of your data. an array of dictionaries for each block is expected')

    if nblocks != len(data): #len(data) should give you the number of blocks in the data file
        raise Exception('there are not as many blocks in the data as you think. check this!')


    for block in data: #iterate over each block of data

        for trace in traces_to_scan:
            if trace not in block.keys():
                raise Exception('the signal %s is missing from your data. check spelling, or check the traces you want to pass to the function!' %trace)
#
#        if 'lx' not in block.keys():
#            raise Exception('A signal relating to the left eye is missing. Make sure the left x is labelled \'lx\'')
#        if 'ly' not in block.keys():
#            raise Exception('A signal relating to the left eye is missing. Make sure the left y is labelled \'ly\'')
#        if 'rx' not in block.keys():
#            raise Exception('A signal relating to the right eye is missing. Make sure the right x is labelled \'rx\'')
#        if 'ry' not in block.keys():
#            raise Exception('A signal relating to the right eye is missing. Make sure the right y is labelled \'ry\'')
#        if 'lp' not in block.keys():
#            raise Exception('The signal relating to the left pupil size is missing. Make sure that the left pupil is labelled \'lp\'')
#        if 'rp' not in block.keys():
#            raise Exception('The signal relating to the right pupil size is missing. Make sure that the right pupil is labelled \'rp\'')

        #create empty vectors to hold start and end points of missing data in the traces specified in function call
        tmpdata = dict()
        for trace in traces_to_scan:
            tmpdata['s_' + trace] = []
            tmpdata['e_' + trace] = []

        #find the missing data in each gaze trace
        for trace in traces_to_scan:
            tmp_mtrace = np.array(np.isnan(block[trace]) == True, dtype = int) #array's of 1/0's if data is missing at a sample
            tmp_dtrace = np.diff(tmp_mtrace) #find the change-points. +1 means goes from present to absent, -1 from absent to present
            tmpdata['s_' + trace] = np.squeeze(np.where(tmp_dtrace ==  1)) #find where it starts to be missing
            tmpdata['e_' + trace] = np.squeeze(np.where(tmp_dtrace == -1)) #find the index of the last missing sample


        for trace in traces_to_scan:
            tmpdata['Eblk_' + trace] = []

        for trace in traces_to_scan:               # loop over all traces
            for i in range(len(tmpdata['s_' + trace])):  # loop over every start of missing data
                if tmpdata['s_' + trace].size == 1: # if only one missing period (unlikely in blocked data, but common if you get trialwise recordings)
                    start = tmpdata['s_' + trace].tolist()
                    end   = tmpdata['e_' + trace].tolist()
                else:
                    start   = tmpdata['s_' + trace][i]
                    if i    < tmpdata['e_' + trace].size: #check within the range of missing periods in the data
                        end = tmpdata['e_' + trace][i]
                    elif i == tmpdata['e_' + trace].size:
                        end = tmpdata['e_' + trace][-1]
                    else:
                        end = tmpdata['e_' + trace][-1] #get the last end point
                ttime_start = block['trackertime'][start]
                ttime_end   = block['trackertime'][end]
                dur = end-start

                # now we'll make a blink event structure
                # blink_code, start (blocktime), end (blocktime), start (trackertime), end (trackertime), duration
                evnt = [trace + '_BLK', start, end, ttime_start, ttime_end, dur]
                tmpdata['Eblk_' + trace].append(evnt)

        #append these new structures to the dataset...
        for trace in traces_to_scan:
            block['Eblk_' + trace] = tmpdata['Eblk_' + trace]
    return data

def interpolateBlinks_Blocked(block, trace):

    """

    example call: block = interpolateBlinks_Blocked(block, trace = 'lx')

    - This function will interpolate the blinks in eyelink data that is in blocks (longer continuous segments)
    - The data that it will handle can either be segmented trials (due to trialwise stop/start of recording in the task) stitched back together
    - or just continuous block data, depending on the structure you give it.

    - If you use the script AttSacc_ParseData.py to parse the data, and are running AttSacc_CleanBlockedData.py, then the block structure should be suitable for this function.

    To interpolate over the blinks present in a trace, the following fields must be present within the data dictionary:
    - trackertime: the time series of the timepoints defined by the eyelink time (rather than trial time!)
    - the trace (e.g. block['lx'])
    - the list of events characterising missing periods of data in that trace (e.g. block['Eblk_lx'])

    these blink structures have the following format:
        ['event_code', start (blocktime), end (blocktime), start (trackertime), end (trackertime), duration]
        event codes e.g. 'lx_BLK'


    Missing periods of data will be removed in the following way:
    periods of missing data of under 10 samples will be linearly interpolated within a window of 10 samples either side of the start and end of the period

    """
    if not isinstance(block, dict):
        raise Exception('data supplied is not a dictionary')

    if not isinstance(trace, str):
        raise Exception('the trace indicator supplied is not a string. \n please input a string e.g. \'lx\'')

    if trace not in block.keys():
        raise Exception('the signal you want to clean does not exist in the data. check your trace labels in your data')

    signal = block[trace] #extract the signal that needs cleaning

    eventlabel = 'Eblk_%s'%trace

    if eventlabel not in block.keys():
        raise Exception('the missing period information for this signal is not in your data structure')

    blinks = np.array(block[eventlabel])       #get the desired blink structures
    if blinks.size != 0:
        blinks = blinks[:,1:]            #remove the first column as it's a string for the event code, not needed now
        blinks = blinks.astype(float).astype(int) # change the strings to integers. need to go via float or it fails.


        short_duration_inds  = np.where(np.in1d(blinks[:,4], range(21)))[0]    # find the blinks that are below 20 samples long
        medium_duration_inds = np.where(np.in1d(blinks[:,4], range(21,51)))[0] # find the blinks that are between 21 and 50 samples long
        long_duration_inds   = np.where(blinks[:,4] > 50)[0]                   # find blinks that are over 50 samples long

        short_blinks  = blinks[short_duration_inds,:]
        medium_blinks = blinks[medium_duration_inds,:]
        long_blinks   = blinks[long_duration_inds,:]

        #linear interpolate across these smaller periods before proceeding.
        for blink in short_blinks:
            start, end               = blink[0], blink[1] #get start and end periods of the missing data
            to_interp                = np.arange(start, end) #get all time points to be interpolated over

            #set up linear interpolation
            inttime                  = np.array([start,end])
            inttrace                 = np.array([signal[start], signal[end]])
            fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')

            interptrace              = fx_lin(to_interp)
            signal[to_interp] = interptrace
        for blink in medium_blinks:
            start, end               = blink[0], blink[1] #get start and end periods of the missing data
            to_interp                = np.arange(start, end) #get all time points to be interpolated over

            #set up linear interpolation
            inttime                  = np.array([start,end])
            inttrace                 = np.array([signal[start], signal[end]])
            fx_lin                   = sp.interpolate.interp1d(inttime, inttrace, kind = 'linear')

            interptrace              = fx_lin(to_interp)
            signal[to_interp] = interptrace

        #now cubic spline interpolate across the larger missing periods (blinks)
        for blink in long_blinks:
            start, end            = blink[0], blink[1] #get start and end of these missing samples
            if end+40 >= signal.size: #this blink happens just before the end of the block, so need to adjust the window
                window            = [start-100, start - 50, end, signal.size-1] #reduce the window size but still cubic spline
            elif end+40 <= signal.size and end+80 >= signal.size:
                window            = [start-100, start-50, end+50, signal.size-1]
            else:
                window            = [start-100, start-50, end+50, end+100] #set the window for the interpolation
            inttime               = np.array(window)
            if end + 50 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50])                 , np.nanmedian(signal[start-50:start-1]),
                                              np.nanmedian(signal[end:int(np.floor((signal.size-1-end)/2))]), np.nanmedian(signal[int(np.ceil((signal.size-1-end)/2)):signal.size-1]) ])
            elif end+50 <= signal.size and end + 100 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50])                 , np.nanmedian(signal[start-50:start-1]),
                                              np.nanmedian(signal[end:end+50]), np.nanmedian(signal[end+50:signal.size-1]) ])
            else:
                inttrace          = np.array([np.nanmedian(signal[start-100:start-50]), np.nanmedian(signal[start-50:start-1]), # by giving the nanmedian between these points,
                                              np.nanmedian(signal[end+1:end+50])     , np.nanmedian(signal[end+50:end+100]) ])  # points, it accounts for variance of the signal
            fx_cub                = sp.interpolate.interp1d(inttime, inttrace, kind = 'cubic')


            if end+50 >= signal.size:
                to_interp         = np.arange(start-50, signal.size-1)
            else:
                to_interp         = np.arange(start-50, end+50) #interpolate just outside the start of the missing period, for cases of large changes due to blinks
            interptrace           = fx_cub(to_interp)
            signal[to_interp]     = interptrace
    #output the data into the block structure
    block[trace] = signal
    return block


def epoch(data, trigger_value, traces, twin = [-.5, +1], srate = 1000, collapse_across_blocks = True):
    """
    this function will epoch your eyetracking data given a specific trigger to find.

    data                    -- this expects a list of dictionaries (the preferred data type for all the functions here), where (in theory, for now), each element of the list is a block of data
    trigger_value           -- this expects a string (as triggers to the eyelink are strings). This function will look for this specific trigger, not other variants containing this trigger
    twin                    -- time window for your epoch. Will default to .5s before, 1s after trigger onset unless specified otherwiseself.
    traces                  -- specify the traces that you want to epoch (e.g. 'lx', 'ly', 'lp', 'rp', etc..)
    srate                   -- the sampling rate of the eyetracker. Defaults to 1000Hz
    collapse_across_blocks  -- boolean for whether you want epochs separated by block, or all epochs in one array

    output:
        this will output a dictionary. Each key within the dictionary will hold a ntriggers x time array of epoched dataself.
        e.g.: output.keys() will give ['lx', 'rx', 'ly', 'ry', 'lp', 'rp']
                output['lx'] will be ntrigs x time array of epochs.
    """

    #check some of the input arguments to make sure we've got the right inputs here
    if not isinstance(trigger_value, str):
        raise Exception('please provide a string for a trigger, not a %s'%type(trigger_value))
    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        raise Exception('an array/list of dictionaries is expected as the input data')
    if not isinstance(traces, list):
        raise Exception('a list of strings is expected for the signal traces you want to epoch')


    output_data = np.array([]) #create empty output data that gets returned at the end
    twin_srate = np.multiply(twin, srate)

    for iblock in range(len(data)): # loop over all blocks of data in the structure
        block = copy.deepcopy(data[iblock])
        trace_dict = dict()
        for trace in traces:
            trace_dict[trace + '_epoch'] = [] #create dummy vars for each trace being operated on
        triggers       = copy.deepcopy(block['Msg'])                                         #get a copy of all the triggers in this block of data
        trig_inds      = [x for x in range(len(triggers)) if triggers[x][2] == trigger_value]
        events_to_find = np.array([triggers[x] for x in trig_inds]) #isolate only the trigger of interests
        epoch_starts   = events_to_find[:,1].astype(float) # and get their starting point in the data (in trackertime)

        triggerinds       = [] # the index of the vectors where the trigger appeared
        for starttime in epoch_starts:
            triggerinds.append(int(np.squeeze(np.where(block['trackertime'] == starttime)))) #append the location (within vector) of the trigger onset

        for trace in traces:
            for triggerstart in triggerinds:
                tmpepoch = block[trace][triggerstart + int(twin_srate[0]):triggerstart + int(twin_srate[1])]
                trace_dict[trace + '_epoch'].append(tmpepoch)
            trace_dict[trace + '_epoch'] = np.array(trace_dict[trace + '_epoch'])
        output_data = np.append(output_data, trace_dict) #add this blocks' epoched data into the data structure

    if collapse_across_blocks:
        output_data = collapse_blocks(output_data, traces = traces, twin_srate = twin_srate)

    return output_data

def collapse_blocks(output_data, traces, twin_srate):
    """
    function to collapse blocked epoched data into one larger array of all blocks instead of epochs within blocks
    needs to know what 'traces' are being used here, but inherits from the epoching function proper
    """

    blocknum = len(output_data) #get the number of blocks
    tmpstruct = dict()

    for trace in traces:
        tmpstruct[trace + '_epoch'] = np.empty([0,int(twin_srate[1]- twin_srate[0])]) #make an empty array of the right shape so we can just vstack all the arrays easily

    for iblock in range(len(output_data)): #loop over all blocks
        for trace in traces: #loop over traces, to make sure we concatenate all epochs made
            tmpstruct[trace + '_epoch'] = np.vstack([tmpstruct[trace + '_epoch'], output_data[iblock][trace + '_epoch']]) # add the new blocks' epochs as new rows in the data

    return tmpstruct
