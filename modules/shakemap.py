import numpy as np
import pandas as pd
from openquake.hazardlib import imt
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.contexts import RuptureContext

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
        rctx.vs30 = sites.vs30
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
        gmm_results = self.gmm.get_mean_and_std(self.rupture, self.stations)
        self.phi = np.mean(gmm_results['phi'])
        self.tau = np.mean(gmm_results['tau'])
        C_SS = self.scm.get_correlation_matrix(self.stations.mesh, full_cov = True)
        jitter_m = np.zeros_like(C_SS)
        np.fill_diagonal(jitter_m, self.jitter) 
        self.inv_C_SS = np.linalg.inv(C_SS + jitter_m)

        recorded_amplitudes = self.stations.get_recordings()[:, None]
        Z = np.ones_like(recorded_amplitudes)

        raw_residuals = recorded_amplitudes - gmm_results['mu_logIM'][:, None]
        self.psiBsq = 1 / ( (1 / self.tau**2) + (Z.T @ (self.inv_C_SS @ Z)) / self.phi**2 )
        self.xiB = self.psiBsq / self.phi**2 * Z.T @ (self.inv_C_SS @ raw_residuals)
        self.residuals = raw_residuals - self.xiB

    def predict_logIM(self, sites, conditional = True, full_cov = True):
        gmm_results = self.gmm.get_mean_and_std(self.rupture, sites)
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
        return rng.multivariate_normal(mean.squeeze(), cov_matrix, nsamples)