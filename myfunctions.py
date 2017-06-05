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
    
def interpolateBlinks(trial):
    x = trial['x']
    y = trial['y']
    time = trial['time']
    blinks = trial['events']['Eblk']
    fsamp = trial['behaviour']['fsamp']
    if len(blinks) ==1:
        blink = blinks[0]
        start, end = blink[0]-fsamp, blink[1]-fsamp