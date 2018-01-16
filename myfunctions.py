# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

author: Sammi Chekroud
"""
import numpy   as np
import scipy   as sp
from scipy import interpolate
import sys
import os

def Eucdist(x1, y1, x2, y2):
    """
    calculate euclidian distance between two points

    formula: sqrt( (x2-x1)^2 + (y2-y1)^2 )    
    
    """
    distance = np.sqrt( (x2-x1)**2 + (y2-y1)**2)
    return distance


def parse_eye_data(eye_fname, block_folder,nblocks, ntrials):
    
    """
    eye_fname expects the full path to a file that needs to be parsed.
        The function will take the name of this, minus the file extension, and create a parsed pickle version
    
    out_folder needs to be a path to where you want the parsed data to be output to.
    
    if you give it os.path.join(some_path, 'fname.asc') it will save:
    os.path.join(block_folder, 'fname_parsed.pickle'). These pickle files are read into later preprocessing functions
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
    #get points where recording stopped
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']
    
    #assign some empty lists to get filled with information
    trackertime = []
    lx   = []; rx   = []
    ly   = []; ry   = []
    lp   = []; rp   = []
    av_x = []; av_y = []
    av_p = []
    
    Efix = []; Sfix = []
    Esac = []; Ssac = []
    Eblk = []; Sblk = []
    Msg  = []
    
    for i in range(len(start_inds)):
        start_line = start_inds[i]
        end_line   = end_inds[i]
        
        itrl = np.array(split_d[start_line:end_line])
        
        itrl_event_inds = [x for x in range(len(itrl)) if itrl[x][0] == 'MSG']        # get the line indices where trigs sent
        itrl_events     = [itrl[x] for x in itrl_event_inds]                              # get the actual triggers
        itrl_fix_inds   = [x for x in range(len(itrl)) if itrl[x][0] == 'EFIX' or itrl[x][0] == 'SFIX']
        itrl_fix        = [itrl[x] for x in itrl_fix_inds]
        itrl_sac_inds   = [x for x in range(len(itrl)) if itrl[x][0] == 'ESACC' or itrl[x][0] == 'SSACC']
        itrl_sac        = [itrl[x] for x in itrl_sac_inds]
        itrl_blink_inds = [x for x in range(len(itrl)) if itrl[x][0] == 'EBLINK' or itrl[x][0] == 'SBLINK']
        itrl_blink      = [itrl[x] for x in itrl_blink_inds]
        
        itrl_nondata    = sorted(itrl_blink_inds + itrl_sac_inds + itrl_fix_inds + itrl_event_inds) #get all non-data line indices
        itrl_data = np.delete(itrl, itrl_nondata) #remove these lines so all you have is the raw data

        itrl_data = itrl_data[6:] #remove first five lines as these are filler after the recording starts        
        
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
        
        #average data across the eyes --- take the nanmean though in case of missing data. we'll still save the independent eyes though as a sanity check
        itrl_x = np.vstack([itrl_lx, itrl_rx])
        itrl_x = np.nanmean(itrl_x, axis = 0)
        itrl_y = np.vstack([itrl_ly, itrl_ry])
        itrl_y = np.nanmean(itrl_y, axis = 0)    
        itrl_p = np.vstack([itrl_lp, itrl_rp])
        itrl_p = np.nanmean(itrl_p, axis = 0)
    
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
        av_x.append(itrl_x)
        av_y.append(itrl_y)
        av_p.append(itrl_p)
        Efix.append(itrl_efix)
        Sfix.append(itrl_sfix)
        Ssac.append(itrl_ssac)
        Esac.append(itrl_esac)
        Sblk.append(itrl_sblk)
        Eblk.append(itrl_eblk)
        Msg.append(itrl_events)    
    
    #rep_inds  = np.repeat(np.arange(nblocks),trialsperblock)
    #plt.hist(rep_inds, bins = nblocks) # all bars should be the same height now if its work? @ trials per block (80)
    
    iblock = {
    'trackertime': [],
    'av_x': [], 'av_y': [], 'av_p': [],
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
            iblock_data['av_x'].append(av_x[ind])
            iblock_data['av_y'].append(av_y[ind])
            iblock_data['av_p'].append(av_p[ind])
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
        blocked_data[block]['av_x']        = np.hstack(blocked_data[block]['av_x']       )
        blocked_data[block]['av_y']        = np.hstack(blocked_data[block]['av_y']       )
        blocked_data[block]['av_p']        = np.hstack(blocked_data[block]['av_p']       )
        blocked_data[block]['lx']          = np.hstack(blocked_data[block]['lx']         )
        blocked_data[block]['ly']          = np.hstack(blocked_data[block]['ly']         )
        blocked_data[block]['lp']          = np.hstack(blocked_data[block]['lp']         )
        blocked_data[block]['rx']          = np.hstack(blocked_data[block]['rx']         )
        blocked_data[block]['ry']          = np.hstack(blocked_data[block]['ry']         )
        blocked_data[block]['rp']          = np.hstack(blocked_data[block]['rp']         )
    return blocked_data




def find_missing_periods(data, nblocks):
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
    '''
    
    if not isinstance(data, np.ndarray):
        raise Exception('check the format of your data. an array of dictionaries for each block is expected')
    
    if nblocks != len(data): #len(data) should give you the number of blocks in the data file
        raise Exception('there are not as many blocks in the data as you think. check this!')
        
    
    for block in data: #iterate over each block of data
        if 'lx' not in block.keys():
            raise Exception('A signal relating to the left eye is missing. Make sure the left x is labelled \'lx\'')
        if 'ly' not in block.keys():
            raise Exception('A signal relating to the left eye is missing. Make sure the left y is labelled \'ly\'')
        if 'rx' not in block.keys():
            raise Exception('A signal relating to the right eye is missing. Make sure the right x is labelled \'rx\'')
        if 'ry' not in block.keys():
            raise Exception('A signal relating to the right eye is missing. Make sure the right y is labelled \'ry\'')
            
        s_lx = []; s_rx = [];
        s_ly = []; s_ry = [];
        e_lx = []; e_rx = [];
        e_ly = []; e_ry = [];

        #find missing data in each gaze trace (left x & y, right x & y) separately for precision
        mlx = np.array(np.isnan(block['lx']) == True, dtype = int) # array of 1's and 0's for if data is missing at that time point
        dlx = np.diff(mlx) #find change-points, +1 means goes from present to absent, -1 from absent to present
        s_lx = np.squeeze(np.where(dlx ==  1)) #find points where it starts to be missing
        e_lx = np.squeeze(np.where(dlx == -1))+1 #find points where it stops being missing
        
        mly = np.array(np.isnan(block['ly']) == True, dtype = int)
        dly = np.diff(mly)
        s_ly = np.squeeze(np.where(dly ==  1))
        e_ly = np.squeeze(np.where(dly == -1))+1
        
        mrx = np.array(np.isnan(block['rx']) == True, dtype = int)
        drx = np.diff(mrx)
        s_rx = np.squeeze(np.where(drx ==  1))
        e_rx = np.squeeze(np.where(drx == -1))+1
        
        mry = np.array(np.isnan(block['ry']) == True, dtype = int)
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
            ttime_start = block['trackertime'][start]
            ttime_end   = block['trackertime'][end]
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
                start = s_ly[i]
                if i < e_ly.size: #check within range of the number of missing periods in data
                    end = e_ly[i]
                elif i == e_ly.size:
                    end = e_ly[-1]
                else:
                    end = e_ly[-1]
            ttime_start = block['trackertime'][start]
            ttime_end   = block['trackertime'][end]
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
                start = s_rx[i]
                if i < e_rx.size: #check within range of the number of missing periods in data
                    end = e_rx[i]
                elif i == e_rx.size:
                    end = e_rx[-1]
                else:
                    end = e_rx[-1]
            ttime_start = block['trackertime'][start]
            ttime_end   = block['trackertime'][end]
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
                start = s_ry[i]
                if i < e_ry.size: #check within range of the number of missing periods in data
                    end = e_ry[i]
                elif i == e_ry.size:
                    end = e_ry[-1]
                else:
                    end = e_ry[-1]
            ttime_start = block['trackertime'][start]
            ttime_end   = block['trackertime'][end]
            dur = end-start
            #create a blink event structure:
            # blink_code, start (blocktime), end (blocktime), start_trackertime, end_trackertime, duration
            evnt = ['RY_BLK', start, end, ttime_start, ttime_end, dur]
            Eblk_ry.append(evnt)
        #append these new structures to the dataset
        block['Eblk_lx'] = Eblk_lx
        block['Eblk_rx'] = Eblk_rx
        block['Eblk_ly'] = Eblk_ly
        block['Eblk_ry'] = Eblk_ry
        
    return data

def interpolateBlinks_Blocked(block, trace):

    """      
    - This function will interpolate the blinks in eyelink data that is in blocks (longer continuous segments)
    - The data that it will handle can either be segmented trials (due to trialwise stop/start of recording in the task) stitched back together
    - or just continuous block data, depending on the structure you give it.
        
    - If you use the script AttSacc_ParseData.py to parse the data, and are running AttSacc_CleanBlockedData.py, then the block structure should be suitable for this function.
    
    block argument will expect a dictionary that has the following fields:
    - trackertime: the timeseries of timepoints defined by the eyelink time, rather than trial time
    - lx      - timeseries of the x value for the left eye
    - ly      - timeseries of the y value for the left eye
    - rx      - timeseries of the x value for the right eye
    - ry      - timeseries of the y value for the right eye
    
    - Eblk_lx - list of events characterising missing periods of data in lx
    - Eblk_rx - list of events characterising missing periods of data in rx
    - Eblk_ly - list of events characterising missing periods of data in ly
    - Eblk_ry - list of events characterising missing periods of data in ry
    ----------- these blink structures have the following format:
    ----------- ['Event_code', start_blocktime, end_blocktime, start_trackertime, end_trackertime, duration]
    ----------- block code e.g. 'LX_BLK'
    
    trace argument expects a string specifying what trace you want to clean (e.g. 'lx' for left eye, x signal)
        
    
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
                window            = [start-80, start - 40, end, signal.size-1] #reduce the window size but still cubic spline
            elif end+40 <= signal.size and end+80 >= signal.size:
                window            = [start-80, start-40, end+40, signal.size-1]
            else:
                window            = [start-80, start-40, end+40, end+80] #set the window for the interpolation
            inttime               = np.array(window)
            if end + 40 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-80:start-40])                 , np.nanmedian(signal[start-40:start-1]),
                                              np.nanmedian(signal[end:int(np.floor((signal.size-1-end)/2))]), np.nanmedian(signal[int(np.ceil((signal.size-1-end)/2)):signal.size-1]) ])
            elif end+40 <= signal.size and end + 80 >= signal.size:
                inttrace          = np.array([np.nanmedian(signal[start-80:start-40])                 , np.nanmedian(signal[start-40:start-1]),
                                              np.nanmedian(signal[end:end+40]), np.nanmedian(signal[end+40:signal.size-1]) ])
            else:
                inttrace          = np.array([np.nanmedian(signal[start-80:start-40]), np.nanmedian(signal[start-40:start-1]), # by giving the nanmedian between these points, 
                                              np.nanmedian(signal[end+1:end+40])     , np.nanmedian(signal[end+40:end+80]) ])  # points, it accounts for variance of the signal
            fx_cub                = sp.interpolate.interp1d(inttime, inttrace, kind = 'cubic')
            
            
            if end+30 >= signal.size:
                to_interp         = np.arange(start-30, signal.size-1)
            else:
                to_interp         = np.arange(start-30, end+30) #interpolate just outside the start of the missing period, for cases of large changes due to blinks
            interptrace           = fx_cub(to_interp)
            signal[to_interp]     = interptrace    
    #output the data into the block structure
    block[trace] = signal
    return block