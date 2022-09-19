# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief: Time stamp
"""

DayL = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
import datetime
def get_TimeStamp_str():
    now = datetime.datetime.now() + datetime.timedelta(hours=2)
    h = int(now.strftime("%H"))
    m = int(now.strftime("%M"))
    ss = int(now.strftime("%S"))
    s = now.strftime("%Y-%m-%d")
    s = str(s)
    if m<10:
        m = '0' + str(m)
    if h>12:
        s += "*%2d_%2s_%2dPM"%(h - 12,m,ss)
    elif h == 12:
        s += "*%2d_%2s_%2dPM"%(h ,m,ss)
    else:
        s += "*%2d_%2s_%2dAM"%(h,m,ss)
    return DayL[now.weekday()]+'_'+s.replace(' ','0').replace('*',' ')
