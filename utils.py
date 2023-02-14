import numpy as np
import pandas as pd
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface.multi import MultiSurface
from openquake.hazardlib.geo.mesh import Mesh

def get_planarsurface(rup_geom_json):
    fs = []
    for rup_geomt in rup_geom_json['coordinates'][0]:
        gtemp = np.array(rup_geomt[:-1])
        unique_dephts = np.sort(np.unique(gtemp[:,2]))
        gtemp_top = list(np.sort(gtemp[gtemp[:,2]==unique_dephts[0], :], axis=0))
        gtemp_bottom = list(np.sort(gtemp[gtemp[:,2]==unique_dephts[1], :], axis=0))

        for i in np.arange(0, len(gtemp_top)-1, 1):
            fs.append(
                PlanarSurface.from_corner_points(top_left = Point(gtemp_top[i][0], gtemp_top[i][1], gtemp_top[i][2]),
                                                top_right = Point(gtemp_top[i+1][0], gtemp_top[i+1][1], gtemp_top[i+1][2]),
                                                bottom_right = Point(gtemp_bottom[i+1][0], gtemp_bottom[i+1][1], gtemp_bottom[i+1][2]),
                                                bottom_left = Point(gtemp_bottom[i][0], gtemp_bottom[i][1], gtemp_bottom[i][2]))
            )
    rupture_surface = MultiSurface(fs)
    return rupture_surface

def read_stationlist(stations):
    stations_t = []
    for station in stations:
        if station['properties']['station_type'] != 'seismic':
            continue
        station_t = {
            'id': station['id'],
            'code': station['properties']['code'],
            'longitude': station['geometry']['coordinates'][0],
            'latitude': station['geometry']['coordinates'][1],
            'vs30': station['properties']['vs30'],
        }
        for direction, channel in zip(['E', 'N'], station['properties']['channels'][:2]):
            if channel['name'][-1] not in ['E', 'N', '1', '2']:
                print('warning! Wrong value associated with one direction')
                print('station code: ' + str(station_t['code']))
                continue
            else:
                for amplitude in channel['amplitudes']:
                    name = amplitude['name'] + '_' + direction
                    val = amplitude['value']
                    if amplitude['name'] not in ['pgv']:
                        val = val / 100 # get g for accelerations
                    station_t[name] = val
        stations_t.append(station_t)
    return pd.DataFrame(stations_t)


class Sites(object):
    def __init__(self, df: pd.DataFrame, column_names: list):
        self.df = df
        self.mesh = Mesh(df[column_names[0]].values, 
                         df[column_names[1]].values, 
                         depths=None)
        self.n_sites = len(df)
        self.vs30 = df[column_names[2]].values

class Stations(Sites):
    def __init__(self, df: pd.DataFrame, column_names: list, im_string: str):
        super().__init__(df, column_names)
        self.im_string = im_string
    def get_recordings(self):
        # computes the geometric mean (average horizontal) of the recorded im
        col = self.im_string.lower()
        recordings = np.sqrt(self.df[col + '_E'] * self.df[col + '_N']).values
        return np.log(recordings)

class GMM(object):
    def __init__(self, gmm_oq, im_string: str):
        self.im = imt.from_string(im_string)
        self.gmm = gmm_oq
    
    def get_mean_and_std(self, rupture, sites):
        n = sites.n_sites
        rctx = RuptureContext()
        rctx.rjb = rupture.surface.get_joyner_boore_distance(sites.mesh)
        rctx.rrup = rupture.surface.get_min_distance(sites.mesh)
        rctx.vs30 = sites.vs30.values
        rctx.mag = rupture.mag
        rctx.rake = rupture.rake

        mean = np.zeros([1, n])
        sigma = np.zeros([1, n])
        tau = np.zeros([1, n])
        phi = np.zeros([1, n])
        self.gmm.compute(rctx, [self.im], mean, sigma, tau, phi)

        return {'mu_logIM': mean[0,:], 'phi': phi[0,:], 'tau': tau[0,:]}     


class Shakemap(object):
    def __init__(self, Rupture, Stations, 
                 GMM, SpatialCorrelationModel, jitter=1e-6):
        self.rupture = Rupture
        self.stations = Stations
        self.gmm = GMM
        self.scm = SpatialCorrelationModel
        self.jitter = jitter
        self._update_residuals()

    def _update_residuals(self):
        gmm_results = self.gmm(self.rupture, self.sites)
        self.phi = np.mean(gmm_results['phi'])
        self.tau = np.mean(gmm_results['tau'])
        C_SS = self.scm.get_correlation_matrix(self.stations.mesh, full_cov = True)
        jitter_m = np.zeros_like(C_SS)
        np.fill_diagonal(jitter_m, self.jitter) 
        self.inv_C_SS = np.linalg.inv(C_SS + jitter_m)

        recorded_amplitudes = self.stations.get_recordings()
        Z = np.ones_like(recorded_amplitudes)

        raw_residuals = recorded_amplitudes - gmm_results['mu_logIM'][:, None]
        self.psiBsq = 1 / ( (1 / self.tau**2) + (Z.T @ (self.inv_C_SS @ Z)) / self.phi**2 )
        self.xiB = self.psiBsq / self.phi**2 * Z.T @ (self.inv_C_SS @ raw_residuals)
        self.residuals = raw_residuals - self.xiB

    def predict_logIM(self, sites, conditional = True, full_cov = True):
        gmm_results = self.gmm(self.rupture, sites)
        C_TT = self.scm.get_correlation_matrix(sites.mesh, full_cov = full_cov)
        if conditional is False:
            mean = gmm_results['mu_logIM'][:, None]
            cov_matrix = self.tau**2 + self.phi**2 * C_TT
        else:
            C_TS = self.scm.get_correlation_matrix(sites.mesh, self.stations.mesh)
            mean = (gmm_results['mu_logIM'][:, None] + self.xiB + 
                    (C_TS @ (self.inv_C_SS @ self.residuals) ) )
            if full_cov:
                cov_matrix = ((self.phi**2 + self.psiBsq) * 
                              (C_TT - C_TS @ (self.inv_C_SS @ C_TS.T ) ) )
            else:
               cov_matrix = ((self.phi**2 + self.psiBsq) * 
                              (C_TT - np.diag(C_TS @ (self.inv_C_SS @ C_TS.T)).reshape(-1,1) ) )
        return mean, cov_matrix
    
    def sample_logIM(self, sites, nsamples, conditional = True, seed = None):
        mean, cov_matrix = self.predict_logIM(sites, conditional=conditional)
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(mean, cov_matrix, nsamples)