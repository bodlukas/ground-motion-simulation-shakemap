#%%
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
path_data = 'data' + os.sep

#%%
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.geo.point import Point
from utils import get_planarsurface

option_rupture = 1 # 1 or 2 as stated above

f = open(path_data + 'rupture.json')
rup_temp = json.load(f)
f.close()

if option_rupture == 1:
    rupture_surface = get_planarsurface(rup_temp['features'][0]['geometry'])
elif option_rupture == 2:
    from utils import get_finite_fault
    rupture_surface = get_planarsurface(get_finite_fault())
rup_temp = rup_temp['metadata']
rupture = BaseRupture(mag = rup_temp['mag'], rake = rup_temp['rake'], 
                    tectonic_region_type = 'Active Shallow Crust', 
                    hypocenter = Point(longitude = rup_temp['lon'], 
                                        latitude = rup_temp['lat'],
                                        depth = rup_temp['depth']),
                    surface = rupture_surface)

#%%
from utils import read_stationlist

f = open(path_data + 'stationlist.json')
stations_temp = json.load(f)
stations_temp = stations_temp['features']
f.close()

dfstations = read_stationlist(stations_temp)
dfstations

#%% Simulate
from modules.shakemap import Stations, GMM, Sites
from modules.spatialcorrelation import (HeresiMiranda2019, 
                                        EspositoIervolino2012esm,
                                        BodenmannEtAl2022)
from openquake.hazardlib.gsim.akkar_2014 import AkkarEtAlRjb2014
from openquake.hazardlib.gsim.cauzzi_2014 import CauzziEtAl2014

# Specify considered intensity measure: 'PGA', 'SA(0.3)', 'SA(1.0)', 'SA(3.0)'
im_string = 'SA(1.0)'

# Specify Ground-Motion model
gmm = GMM(CauzziEtAl2014(), im_string)
# gmm = GMM(AkkarEtAlRjb2014(), im_string)

# Specify Correlation model
scm = EspositoIervolino2012esm(im_string)
# scm = HeresiMiranda2019(im_string)
# scm = BodenmannEtAl2022(im_string, rupture)

#%% Simulate
sites = Sites(dfstations)
gmm_results = gmm.get_mean_and_std(rupture, sites)
corr_mat = scm.get_correlation_matrix(sites, full_cov=True)

mean = gmm_results['mu_logIM']
# cov_mat = np.mean(gmm_results['tau'])**2 + np.mean(gmm_results['phi'])**2 * corr_mat
rng = np.random.default_rng(91)
deltaW_sim = (rng.multivariate_normal(np.zeros_like(mean), corr_mat, 1).squeeze() *
               np.mean(gmm_results['phi']) )
deltaB_sim = rng.normal() * np.mean(gmm_results['tau'])
sim = mean + deltaW_sim + deltaB_sim
# sim = rng.multivariate_normal(mean, cov_mat, 1).squeeze()

#%%
dfstations_sim = dfstations.copy()
dfstations_sim['sa(1.0)_E'] = np.exp(sim)
dfstations_sim['sa(1.0)_N'] = np.exp(sim)


#%%
stations = Stations(dfstations_sim, im_string)
from modules.shakemap import Shakemap
shakemap = Shakemap(Rupture = rupture, 
                    Stations = stations,
                    GMM = gmm,
                    SCM = scm)

# %%
plt.plot(shakemap.residuals.squeeze() - deltaW_sim, '.k')
# %%
