import os, sys, traceback, pickle
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from ..pkl_utils import dump
from ..time_utils import get_TimeStamp_str
from .Sensor import Sensor
from .Buoy import Buoy
from .UN_utils import calc_distance, random_float
from .positions import get_grid_map_positions, buoys_centers, buoy_neighbour_dict
import time
import pylab as pl
from IPython import display
import random

class Simulation:
    def __init__(self,
                 num_buoys = 3,
                 num_sensors_per_buoy=5,
                 buoys_radius = 0.1,
                 boundries_threshold=0.10,
                 simulation_period = 200,
                 real_time_tracking = False,
                 buoys_data = [],
                 x_lim=(0.2, 0.8),
                 y_lim=(0.2, 0.8),
                 sleep = 0.2,
                 get_tracking_info = False,
                 plt_text = False,
                 scaling_coef =1,
                 show_plt_axis = False,
                 signal_strength_parms_dic = {},
                 dump_initial_simulation_experiment = False,
                 ):
        a = time.time()
        self.num_sensors_per_buoy = num_sensors_per_buoy
        self.num_buoys = num_buoys
        self.buoys_radius = buoys_radius
        self.sensor_matrix = np.empty((0,3), dtype=float, order='C')
        # self.buoy_positions = []
        self.sleep = sleep
        self.img = None
        self.boundries_threshold = boundries_threshold
        self.get_tracking_info=get_tracking_info

        self.show_plt_axis = show_plt_axis
        self.plt_text=plt_text
        self.scaling_coef = scaling_coef
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.tick = 0
        self.simulation_period = simulation_period
        self.sensor_dict = {}    # dictionary of Sensor objects
        self.buoys_dict = {}     # array of Buoy   objects
        self.log = 'Log %s'%get_TimeStamp_str()
        self.real_time_tracking = real_time_tracking
        self.hand_over_matrix = []

        self.buoys_centers = buoys_centers
        self.buoy_neighbour_dict = buoy_neighbour_dict
        self.ML_df = None
        self.env_data_details = []

        self.initialize_Simulation()
        self.read_Environmental_data(buoys_data)

        self.sig_strength_val = -1
        self.signal_strength_parms_dic = signal_strength_parms_dic
        self.calc_sig_strength_parm()

        if dump_initial_simulation_experiment:
            self.dump_Simulation_exp(exp_name='Initial_exp')
        print('Initialization_time:%-.2f Seconds'%(time.time()-a))


    def calc_sig_strength_parm(self):
        self.sig_strength_val = (self.signal_strength_parms_dic['p_t']*\
                                self.signal_strength_parms_dic['g_t']*\
                                self.signal_strength_parms_dic['g_r']*\
                                self.signal_strength_parms_dic['lambda']**2)/((4*np.pi)**2)

    def get_handover_df(self, dump=False):
        self.hand_over_matrix = np.array(self.hand_over_matrix)
        if len(self.hand_over_matrix)==0:
            print('Empty')
            return
        self.ML_df = pd.DataFrame(self.hand_over_matrix,
                                  columns=['time_tick', 'sensor_id',
                                           'sensor_x_pos', 'sensor_y_pos', 'buoy_id', 'handover_buoy_id',
                                           'UW_speed', 'UW_direction']+
                                           ['Sig_0'] +
                                           ['Sig_%d'%i for i in range(1,7)])

        def get_buoy_center(b_id):
            return self.buoys_centers[int(b_id) - 1]
        def get_handover(raw):
            return self.buoy_neighbour_dict[raw['buoy_id']].index(raw['handover_buoy_id']) + 1


        self.ML_df['buoy_x'], self.ML_df['buoy_y'] = zip(*self.ML_df['buoy_id'].apply(lambda x: get_buoy_center(x)))
        self.ML_df['sensor_x_orgin'] = self.ML_df['sensor_x_pos'] - self.ML_df['buoy_x']
        self.ML_df['sensor_y_orgin'] = self.ML_df['sensor_y_pos'] - self.ML_df['buoy_y']
        self.ML_df['y'] = self.ML_df[['buoy_id', 'handover_buoy_id']].apply(lambda raw: get_handover(raw), axis=1)

        for col in ['buoy_id','time_tick','sensor_id','buoy_id','handover_buoy_id']:
            self.ML_df[col] = self.ML_df[col].astype(int)

        if dump:
            self.dump_ML_df()
        return self.ML_df

    def dump_ML_df(self, df_name=''):
        if self.ML_df is not None:
            self.ML_df.to_csv('Simulation_exp/%s_ML_df%s.csv'%(self.log, ' '+df_name))
            print('successfully dumped to "Simulation_exp/%s"'%('%s_ML_df.csv'%self.log), '  Shape: %d'%self.ML_df.shape[0])

    def read_Environmental_data(self, buoys_data):
        buoys_data = buoys_data[:self.num_buoys]

        min_ = 1000000
        for buoy_id in sorted(self.buoys_dict.keys()):
            df = pd.read_csv(buoys_data[buoy_id-1], usecols=['Surface flow velocity','Surface frankincense'])
            self.env_data_details.append([buoy_id,
                                          *[df['Surface flow velocity'].min(),df['Surface flow velocity'].max(),df['Surface flow velocity'].mean()],
                                          *[df['Surface frankincense'].min(), df['Surface frankincense'].max(),df['Surface frankincense'].mean()],
                                          ])

            self.buoys_dict[buoy_id].df = df.values
            if df.shape[0]<min_:
                min_ = df.shape[0]

        self.env_data_details = pd.DataFrame(np.array(self.env_data_details),
                                             columns=['Buoy_id',
                                                      's_min', 's_max', 's_mean',
                                                      'd_min', 'd_max', 'd_mean']
                                             )
        if self.simulation_period == -1:
            self.simulation_period = min_
        else:
            self.simulation_period = min(min_, self.simulation_period)

    def update_Simulation_parameters(self, tracking, sleep, plt_text, signal_strength_parms_dic):
        # in case of running a loaded simulation experiment using different exp-parameters
        if tracking is not None:
            self.real_time_tracking = tracking

        if sleep is not None:
            self.sleep = sleep

        if plt_text:
            self.plt_text = True

        if signal_strength_parms_dic is not None:
            self.signal_strength_parms_dic = signal_strength_parms_dic
            self.calc_sig_strength_parm()

    def run(self, tracking=None, sleep=None, plt_text=False, signal_strength_parms_dic=None):
        print('\n',self.log,'\n'*2)
        self.update_Simulation_parameters(tracking, sleep, plt_text, signal_strength_parms_dic)
        a = time.time()
        for _ in range(self.simulation_period):
            self.tick += 1
            self.update_sensors_pos() # self.df.values[self.tick])
            if self.real_time_tracking:
                self.plot_Simulation_map()
                time.sleep(self.sleep)
            else:
                ...

            print("\r",'==> i: %d  %-.2f%%  sensor_lst:%d' % (self.tick, self.tick * 100 / self.simulation_period, len(self.sensor_dict)),
                      end="")

            if len(self.sensor_dict)==0:
                break

        print('\n\ni: %d  %-.2f%%  sensor_lst:%d' % (self.tick, self.tick * 100 / self.simulation_period, len(self.sensor_dict)),
              '\n\nRuntime: %-.2f Seconds'%(time.time()-a))


        df = self.get_handover_df(dump=True)
        self.dump_Simulation_exp()

        dd = df[['Sig_1', 'Sig_2', 'Sig_3', 'Sig_4', 'Sig_5', 'Sig_6']].copy()
        w = dd.values
        dd['ind'] = np.argmax(w, axis=1) + 1
        df['ind'] = dd['ind'].copy()

        def calc(x):
            return x['ind'] == x['y']

        df['res'] = df[['ind', 'y']].apply(calc, axis=1)
        print((df['res'].sum() + 0.0) / df.shape[0], '%')

    def update_sensors_pos(self):
        for s_key in sorted(self.sensor_dict.keys()):
            s_,d_ = self.buoys_dict[self.sensor_dict[s_key].buoy_id].df[self.tick-1]
            self.sensor_dict[s_key].update_curr_pos(  s_=s_,
                                                      d_=d_,
                                                      alfa=1,
                                                      scaling_coef=self.scaling_coef
                                                   )

            last_buoy_id = self.sensor_dict[s_key].buoy_id
            new_buoy_id, least_dis, strength_lst = self.get_buoy_using_signal_strength(self.sensor_dict[s_key])

            if self.get_tracking_info:
                print(self.tick, ': sensor_id:%d' % self.sensor_dict[s_key].id, '(%-02f, %-02f)' % (self.sensor_dict[s_key].pos[0],
                      self.sensor_dict[s_key].pos[1]),
                      last_buoy_id, '=>', new_buoy_id, '',
                      'Dist:%-.3f' % least_dis, 'sensor_lst:%d' % len(self.sensor_dict), ' C:', self.sensor_dict[s_key].color)

            if new_buoy_id==-5:
                # self.write_log_file('\n\n====================', self.sensor_dict[s_key].id, 'dist:%-.3f'%least_dis,
                #                     'is out ===============================================================')
                # self.write_log_file(self.tick, '-s_ID:%d'%self.sensor_dict[s_key].id, '_B:%d'%self.sensor_dict[s_key].buoy_id, '\n',
                #                     'Last:', self.sensor_dict[s_key].last_sig_strength, '\n',
                #                     'Acct:', strength_lst, '\n')
                self.sensor_matrix[self.sensor_dict[s_key].id-1, 2] = 1  # red_color
                del self.sensor_dict[s_key]
                # self.sensor_lst.remove(s_key)
            else:
                self.sensor_dict[s_key].update_buoy(new_buoy_id)
                self.sensor_matrix[self.sensor_dict[s_key].id-1, 0] = self.sensor_dict[s_key].x
                self.sensor_matrix[self.sensor_dict[s_key].id-1, 1] = self.sensor_dict[s_key].y
                self.sensor_matrix[self.sensor_dict[s_key].id-1, 2] = self.sensor_dict[s_key].buoy_id # new sector color

            if new_buoy_id != last_buoy_id and new_buoy_id != -5:
                # self.write_log_file('\n'*2, self.tick, '-', self.sensor_dict[s_key].id, '_', self.sensor_dict[s_key].buoy_id, '\n',
                #       self.sensor_dict[s_key].last_sig_strength, '\n', strength_lst, '\n')
                self.hand_over_matrix.append(
                                                [self.tick, self.sensor_dict[s_key].id,
                                                 *self.sensor_dict[s_key].pos_history[-2], new_buoy_id,
                                                 *self.buoys_dict[self.sensor_dict[s_key].buoy_id].df[self.tick-2],
                                                 *self.sensor_dict[s_key].last_sig_strength]
                                            )

            try:
                # to handle it this sensor is deleted
                self.sensor_dict[s_key].last_sig_strength = strength_lst
            except:
                ...

    def get_buoy_using_signal_strength(self, s):
        last_buoy = s.buoy_id
        new_buoy_id, highest_strength, least_dis = -1, -100, 100
        neighbours = buoy_neighbour_dict[last_buoy]

        strength_lst = []
        for buoy_id in [last_buoy, *neighbours]:
            if buoy_id == -1:
                strength_lst.append(-1)
            else:
                dist = calc_distance(s.pos, self.buoys_dict[buoy_id].pos)
                ss = self.sig_strength_val/dist**2 # signal strength

                if ss > highest_strength:
                    new_buoy_id = buoy_id
                    least_dis = dist
                    highest_strength = ss

                strength_lst.append(np.round(ss, 4))

        if least_dis > self.boundries_threshold:  # the sensor got out of the buoys
            new_buoy_id = -5

        return new_buoy_id, least_dis, strength_lst

    def initialize_Simulation(self, save_initial_map_with_sensor=False):
        '''
        Create Soensors and Buoys, and distrubute sensors positions over their belonging Buoys
        '''
        try:
            os.makedirs('../Simulation_exp')
        except:
            ...
        print('..........................')
        plt.rcParams["figure.figsize"] = [10, 10]
        fig, ax = plt.subplots()

        def label(xy, text,font_size=10):
            plt.text(x=xy[0] - 0.001,
                     y=xy[1] - 0.02,
                     s=text,
                     ha="center", family='sans-serif',
                     size=font_size
                     )

        buoy_positions = get_grid_map_positions(self.num_buoys) #self.num_buoys)
        patches = []

        bound = self.buoys_radius/2 + 0.01
        for buoy_id, g in enumerate(buoy_positions, 1):
            self.buoys_dict[buoy_id] = Buoy( x=g[0],
                                             y=g[1],
                                             buoy_id=buoy_id,
                                             neighbours_buoys=buoy_neighbour_dict[buoy_id]
                                           )


            polygon = mpatches.RegularPolygon(g, 6, 0.1)
            label(g, str(buoy_id))

            if buoy_id >= 1:
                loc = random_float(number=self.num_sensors_per_buoy,
                                   x_boundaries=(g[0] - bound, g[0] + bound),
                                   y_boundaries=(g[1] - bound, g[1] + bound))
                self.sensor_matrix = np.vstack((
                                                self.sensor_matrix,
                                                np.hstack((
                                                            loc,
                                                            np.array([buoy_id]*self.num_sensors_per_buoy).reshape(-1,1)
                                                          ))
                                               ))
                for id, (x_,y_) in enumerate(loc):
                    s_id = len(self.sensor_dict)+1
                    self.sensor_dict[s_id] = Sensor( id=s_id,
                                                     x = x_,
                                                     y = y_,
                                                     buoy_id = buoy_id
                                                   )
            patches.append(polygon)

        colors = np.linspace(0, 0.99, len(patches) + 2)
        random.shuffle(colors)
        collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
        collection.set_array(colors)
        ax.add_collection(collection)

        pl.xlim(self.x_lim)
        pl.ylim(self.y_lim)
        plt.axis('off')
        plt.savefig('Simulation_exp/%s.png'%self.log, bbox_inches='tight') # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html

        if save_initial_map_with_sensor:
            pl.scatter(self.sensor_matrix[:, 0], self.sensor_matrix[:, 1],
                       s=3,
                       c='blue',  # self.sensor_matrix[:, 2]/len(self.buoys_dict),
                       alpha=0.5
                       )
            plt.savefig('%s.pdf'%self.log, dpi=1000, bbox_inches='tight')
            plt.show()

        self.img = plt.imread('Simulation_exp/%s.png' % self.log)

        os.remove('Simulation_exp/%s.png' % self.log)
        plt.close()

    def plot_Simulation_map(self, time_point = None):
        '''
        plots the simulation map at any specific time point
        '''

        pl.clf()
        if self.show_plt_axis==False:
            pl.axis('off')
        pl.imshow(self.img, extent=[*self.x_lim, *self.y_lim])

        pl.xlim(self.x_lim)
        pl.ylim(self.y_lim)
        pl.scatter(self.sensor_matrix[:, 0], self.sensor_matrix[:, 1],
                   s=7,
                   c='blue',# c=self.sensor_matrix[:, 2]/len(self.buoys_dict),
                   alpha=0.5
                   )
        s = ''
        try:
            if self.plt_text==True:
                for s_key in self.sensor_dict.keys():
                    s += str(self.tick) + ' - ' + self.sensor_dict[s_key].get_data() + '\n' #+ self.sensor_lst[1].get_data()
                pl.text(x=0, y=0.5, s=s,
                        fontsize=15, color='Black')
        except:
            print('errorrrrrrrrrr')

        display.display(pl.gcf())  # ,axis='off')

        display.clear_output(wait=True)

    def dump_Simulation_exp(self, exp_name=''):
        try:
            path = 'Simulation_exp/%s'%(self.log)+ '_%s.pkl'%exp_name
            if not os.path.isdir('Simulation_exp'):
                os.mkdir('Simulation_exp')
            with open( path , 'wb') as f:
                dump(path, self)
            print('successfully dump to "Simulation_exp/%s"'% path) # self.write_log_file
        except FileNotFoundError:
            print('\n**** Err_Z001 Location not found:\n', traceback.format_exc())
        except Exception as exc:
            print('\n**** Err_Z002 Dumping Experiment Err:\n', traceback.format_exc())

    def write_log_file(self, *args):
        try:
            s = ''
            for arg in args: s += str(arg) + ' '
            file = 'Simulation_exp/%s.txt' % (self.log)
            if s.strip() != '':
                with open(file, 'a') as myfile:
                    myfile.write(s + '\n')
                    # print(s)
        except FileNotFoundError:
            os.mkdir('exp')
        except Exception as exc:
            print('\n*** Err_Z003 Writting to file Err:\n', traceback.format_exc())
            sys.exit()


