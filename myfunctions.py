# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

author: Sammi Chekroud
"""
import numpy   as np
import scipy   as sp
from scipy import interpolate


def Eucdist(x1, y1, x2, y2):
    """
    calculate euclidian distance between two points

    formula: sqrt( (x2-x1)^2 + (y2-y1)^2 )    
    
    """
    distance = np.sqrt( (x2-x1)**2 + (y2-y1)**2)
    return distance
    
def interpolateBlinks(trial, kind = 'cubic'):
    
    """
    interpolate bad data due to the presence of a blink
    loops through blinks identified by the eyetracker, and interpolates across them
    
    trial         = trial dictionary
                    (each participant has a list of dictionaries, where each item in the list is a trial, and the dict contains all info)  

    kind          = 'cubic', 'linear'
                    the kind of interpolation you want to use. default is 'cubic' to do cubic spline interpolation
    interpolate 15 samples before and after blink as margin
    
    if a blink happens at the start/end of a trial then we won't interpolate. If at the start, replace missing data with the average of the 40ms after the end of the blink. If at the end, replace missing data with the average of the 40ms before the blink.
    Interpolation method breaks down at these points because missing values are set to 0, not 1, and it fits a line to data that are 0 not NaN, messing with the interpolation
    """
    x      = trial['x']
    y      = trial['y']
    time   = trial['time']
    blinks = trial['events']['Eblk']
    fstime  = trial['trackertime'][0]
    if len(blinks) == 0:
        pass
    else:
        for blink in blinks:
            start, end = blink[0]-fstime, blink[1]-fstime #get start/end times of blink relative to trial start. this is an underestimate (~10ms) from the eyetracker
            
            if end+80 >= len(time): #check if blink happened across end of the trial (i.e. intend beyond last trial sample)
                intstart, intend = start-80, end+80       #take points around the blink to use in the interpolation function
                intend = time[-1] #set intend to end of trial
                to_interp = np.arange(start-80, intend)
                x[to_interp] = np.nanmean(x[start-80:start-40]) #replace with average of 40 samples prior to blink
                y[to_interp] = np.nanmean(y[start-80:start-40])
            elif start-80 <= time[0]: #blink happened before start of trial/continued into trial start
                intstart, intend = time[0], end + 80       #take points around the blink to use in the interpolation function
                to_interp = np.arange(intstart, end+80)
                x[to_interp] = np.nanmean(x[end+40:end+80])
                y[to_interp] = np.nanmean(y[end+40:end+80])
                
            else: #blink within extremes of trial
                intstart, intend = start-80, end + 80       #take points around the blink to use in the interpolation function
                inttime = np.array([intstart, start-50, end+50+1, intend]) #array of time points used for the interpolation
                intx    = np.array([x[intstart], x[start-50], x[end+50+1], x[intend]])
                inty    = np.array([y[intstart], y[start-50], y[end+50+1], y[intend]])

                fx = sp.interpolate.interp1d(inttime, intx, kind)
                fy = sp.interpolate.interp1d(inttime, inty, kind)

                to_interp = np.arange(start-40, end+40+1)        

                interpolated_x = fx(to_interp)
                interpolated_y = fy(to_interp)

                x[to_interp] = interpolated_x
                y[to_interp] = interpolated_y
                del(fx, fy)
            trial['x'] = x
            trial['y'] = y