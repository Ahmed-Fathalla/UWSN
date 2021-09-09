# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief: Sensor Class
"""

import numpy as np

class Sensor:
    def __init__(self, id, x=0, y=0, buoy_id=-1):
        '''
        :arg
            x: latedit
            y: longtude
        '''
        self.id=id
        self.x = x
        self.y = y
        self.color = buoy_id
        self.pos = [x, y]
        self.buoy_id = buoy_id
        self.pos_history = []
        self.last_sig_strength = [-1]*7

        self.append_curr_pos()

    def set_last_sig_strength(self, l):
        self.last_sig_strength = l

    def append_curr_pos(self):
        self.pos_history.append([*np.round(self.pos, 3), self.buoy_id]) # self.buoy_id

    def get_pos(self):
        return self.pos # self.x, self.y

    def get_pos_history(self):
        return self.pos_history

    def get_data(self):
        return 'ID:%d - Buoy_ID:%d - Pos:(%-.2f,%-.2f)'%(self.id, self.buoy_id, self.pos[0], self.pos[1])

    def update_curr_pos(self, s_=0, d_=0, alfa=1, scaling_coef=1):
        '''
            s: is Speed of underwater sensor node.
            d: is the direction
            alfa: is the ....
            scaling_coef: to scale the movement distance according to the sector's coverage area
        '''
        # self.x += (np.random.random() - 0.4) /50
        # self.y += (np.random.random() - 0.4) /50
        #       s_ * np.sin(d_ * np.pi / 180) * alfa * scaling_coef, s_*np.cos(d_*np.pi/180)*alfa*scaling_coef  )
        self.x += s_*np.sin(d_*np.pi/180)*alfa*scaling_coef # *time
        self.y += s_*np.cos(d_*np.pi/180)*alfa*scaling_coef # *time
        self.pos = [self.x, self.y]
        self.append_curr_pos()

    def update_buoy(self, buoy_id):
        self.buoy_id = buoy_id
        self.pos_history[-1][-1] = buoy_id
        self.color = buoy_id/20

