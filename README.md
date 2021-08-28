# A Robust UWSN Handover Prediction System Using Ensemble Learning [(Go-to-Paper)](https://www.mdpi.com/1424-8220/21/17/5777) [(Download-PDF)](https://www.mdpi.com/1424-8220/21/17/5777/pdf)

## Abstract
The use of underwater wireless sensor networks (UWSNs) for collaborative monitoring and marine data collection tasks is rapidly increasing. One of the major challenges associated with building these networks is handover prediction; this is because the mobility model of the sensor nodes is different from that of ground-based wireless sensor network (WSN) devices. Therefore, handover prediction is the focus of the present work. There have been limited efforts in addressing the handover prediction problem in UWSNs and in the use of ensemble learning in handover prediction for UWSNs. Hence, we propose the simulation of the sensor node mobility using real marine data collected by the Korea Hydrographic and Oceanographic Agency. These data include the water current speed and direction between data. The proposed simulation consists of a large number of sensor nodes and base stations in a UWSN. Next, we collected the handover events from the simulation, which were utilized as a dataset for the handover prediction task. Finally, we utilized four machine learning prediction algorithms (i.e., gradient boosting, decision tree (DT), Gaussian naive Bayes (GNB), and K-nearest neighbor (KNN)) to predict handover events based on historically collected handover events. The obtained prediction accuracy rates were above 95%. The best prediction accuracy rate achieved by the state-of-the-art method was 56% for any UWSN. Moreover, when the proposed models were evaluated on performance metrics, the measured evolution scores emphasized the high quality of the proposed prediction models. While the ensemble learning model outperformed the GNB and KNN models, the performance of ensemble learning and decision tree models was almost identical.

## Running an Experiment
```python
from time import time

from utils.UN_simulator.Simulation import Simulation
from utils.UN_simulator.positions import buoys_centers, buoy_neighbour_dict

lst = glob('Buoy_data/*.csv')
lst.append(lst[0])

# Parameters presented in "Table 1" for calculating "Eq. 7"
signal_strength_parms_dic = {   
                                'p_t':5.  ,
                                'g_t':1  ,
                                'g_r':1  ,
                                'lambda':0.125  
                             }
                                
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
e.run()
    ```
## Citing

If you use the proposed simulation in your work, please cite the accompanying [paper]:

```bibtex
@Article{s21175777,
AUTHOR = {Eldesouky, Esraa and Bekhit, Mahmoud and Fathalla, Ahmed and Salah, Ahmad and Ali, Ahmed},
TITLE = {A Robust UWSN Handover Prediction System Using Ensemble Learning},
JOURNAL = {Sensors},
VOLUME = {21},
YEAR = {2021},
NUMBER = {17},
ARTICLE-NUMBER = {5777},
URL = {https://www.mdpi.com/1424-8220/21/17/5777},
ISSN = {1424-8220},
DOI = {10.3390/s21175777}
}
```
[paper]: https://www.mdpi.com/1424-8220/21/17/5777
