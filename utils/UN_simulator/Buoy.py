# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg - a.fathalla_sci@yahoo.com>
@brief: Buoy Class
"""

class Buoy:
    def __init__(self, x=-9999, y=-9999, buoy_id=-1, neighbours_buoys = [], df=None):
        self.x = x
        self.y = y
        self.pos = [x, y]
        self.buoy_id = buoy_id
        self.df = df
        self.neighbors_buoys = neighbours_buoys

    def get_pos(self):
        return self.x, self.y

    def set_pos(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
