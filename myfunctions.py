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
    """    
    x      = trial['x']
    y      = trial['y']
    blinks = trial['events']['Eblk']
    fstime  = trial['trackertime'][0]
    if len(blinks) ==1:
        blink = blinks[0] #only one blink
        start, end = blink[0]-fstime, blink[1]-fstime #get start/end times of blink relative to trial start. this is an underestimate (~10ms) from the eyetracker
        intstart, intend = start-30, end + 30       #take points around the blink to use in the interpolation function
        
        inttime = np.array([intstart, start, end, intend]) #array of time points used for the interpolation
        intx    = np.array([x[intstart], x[start], x[end], x[intend]])
        inty    = np.array([y[intstart], y[start], y[end], y[intend]])
        
        fx = sp.interpolate.interp1d(inttime, intx, kind)
        fy = sp.interpolate.interp1d(inttime, inty, kind)
        
        to_interp = np.arange(start-15, end+15+1)        
        
        interpolated_x = fx(to_interp)
        interpolated_y = fy(to_interp)

        x[to_interp] = interpolated_x
        y[to_interp] = interpolated_y
    trial['x'] = x
    trial['y'] = y






