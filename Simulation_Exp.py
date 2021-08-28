import numpy as np
import pandas as pd
from glob import glob
import os
from time import time

from utils.UN_simulator.Simulation import Simulation
from utils.UN_simulator.positions import buoys_centers, buoy_neighbour_dict

lst = glob('Buoy_data/*.csv')
lst.append(lst[0])

signal_strength_parms_dic = {   'p_t':1  ,
                                'g_t':1  ,
                                'g_r':1  ,
                                'c':1  ,
                                'f':1  }
                                
e = Simulation(
                num_sensors_per_buoy=10,
                num_buoys= len(buoys_centers),
                buoys_radius = 0.1,
                sleep = 0.1,
                simulation_period = -1,
                x_lim=(-0.5, 1.5),
                y_lim=(-0.5, 1.5),
                plt_text = False,
                real_time_tracking = False,
                buoys_data = lst ,#['data/2014.csv'],
                scaling_coef =0.0001,
                show_plt_axis = False,
                signal_strength_parms_dic=signal_strength_parms_dic,
                dump_initial_simulation_experiment=0
               )
# e.plot_Simulation_map()
a = time()
e.run(0)
print('Run_function NB:%-.2f second'%(time()-a))

df = e.get_handover_df(dump=True)