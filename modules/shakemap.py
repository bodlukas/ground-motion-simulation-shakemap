import numpy as np
import pandas as pd
import warnings
from openquake.hazardlib import imt
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.contexts import RuptureContext
from openquake.hazardlib.source.rupture import BaseRupture

valid_im_strings = ['PGA', 'SA(0.3)', 'SA(1.0)', 'SA(3.0)']

class Sites(object):
    """Wrapper for geographic sites at which to predict ground-motion amplitudes.

    Attributes:
        mesh (openquake.Mesh): A mesh instance of the sites that is used 
            for distance computations in openquake.
        
        n_sites (int): Number of sites

        vs30 (np.array): Vs30 values of the sites in m/s
    """    
    def __init__(self, df: pd.DataFrame, 
                 column_names: list = ['longitude', 'latitude', 'vs30']):
        """Initialize Sites object

        Args:
            df (pd.DataFrame): Sites at which to predict ground-motion amplitudes. 
                Each row is a specific site. The dataframe should contain 
                the three columns specified in column_names (see below). 

            column_names (list, optional): Indicate column names in following order: 
                First entry: longitude in degrees / Second entry: latitude in degrees
                Third entry: vs30 in m/s. Defaults to ['longitude', 'latitude', 'vs30'].
        """        
        self.mesh = Mesh(df[column_names[0]].values, 
                         df[column_names[1]].values, 
                         depths=None)
        self.n_sites = len(df)
        self.vs30 = df[column_names[2]].values

    def get_bbox(self):
        """Computes the bounding box of the sites for plotting purposes

        Returns:
            longitudes, latitudes (tuple): Bounding box polygon
                that can be used like: plt.plot(longitudes, latitudes)
        """        
        bbox = [self.mesh.lons.max(), self.mesh.lons.min(), 
                self.mesh.lats.max(), self.mesh.lats.min()]
        lons_bbox = [bbox[0], bbox[0], bbox[1], bbox[1], bbox[0]]
        lats_bbox = [bbox[2], bbox[3], bbox[3], bbox[2], bbox[2]]
        return np.array(lons_bbox), np.array(lats_bbox)

class Stations(Sites):
    """Wrapper for seismic stations (builds on Sites)
    """    
    def __init__(self, df: pd.DataFrame, im_string: str, **kwargs):
        """Initialize Stations object

        Args:
            df (pd.DataFrame): Information extracted from stationlist.json 
                as provided by the USGS. See utils.read_stationlist and 
                shakemap.Sites for further information. 
            im_string (str): Indicate the im for which to extract recordings.
                USGS currently provides 'PGA', 'SA(0.3)', 'SA(1.0)', 'SA(3.0)'.
        """        
        super().__init__(df, **kwargs)
        if im_string not in valid_im_strings:
            raise ValueError(f'im_string must be one of {valid_im_strings}')
        self.im_string = im_string
        self.df = df
    
    def get_recordings(self):
        '''Computes the geometric mean (average horizontal) of the recorded im

        The column names that contain the two horizontal directions builds on 
        name conventions of the USGS stationlist.json file and should end with 
        E and N, respectively. For example: 'sa(1.0)_E' and 'sa(1.0)_N'.

        Adapt this if you choose a ground-motion model which was derived for 
        another metric (such as RotD50).

        Returns:
            log_recordings (np.array): Logarithmic geometric mean of recorded im
        '''
        # computes the geometric mean (average horizontal) of the recorded im
        col = self.im_string.lower()
        recordings = np.sqrt(self.df[col + '_E'] * self.df[col + '_N']).values
        return np.log(recordings)

class GMM(object):
    """Wrapper for an openquake ground-motion model (GMM)
    """    
    def __init__(self, gmm_oq, im_string: str):
        """Initialize GMM wrapper

        Args:
            gmm_oq (openquake.gsim): Openquake GMM
            im_string (str): im to predict
        """        
        self.im = imt.from_string(im_string)
        self.gmm = gmm_oq
    
    def get_mean_and_std(self, rupture: BaseRupture, sites: Sites):
        """Predicts mean and standard deviations of im at sites

        Args:
            rupture (BaseRupture): Openquake rupture instance
            sites (Sites): Sites for which to predict im

        Returns:
            results (dict): Dictionary with logarithmic mean 
                ('mu_logIM'), and std dev of within-event ('phi') 
                and between event ('tau') residuals. Each is an
                array of size (n_sites,).
        """        
        n = sites.n_sites
        # Specify RuptureContext for openquake
        rctx = RuptureContext()
        rctx.rjb = rupture.surface.get_joyner_boore_distance(sites.mesh)
        rctx.rrup = rupture.surface.get_min_distance(sites.mesh)
        rctx.vs30 = sites.vs30
        rctx.mag = rupture.mag
        rctx.rake = rupture.rake

        # Initialize output arrays
        mean = np.zeros([1, n]); sigma = np.zeros([1, n])
        tau = np.zeros([1, n]); phi = np.zeros([1, n])
        self.gmm.compute(rctx, [self.im], mean, sigma, tau, phi)

        return {'mu_logIM': mean[0,:], 'phi': phi[0,:], 'tau': tau[0,:]}     


class Shakemap(object):
    """Main object for shakemap computations and simulations.
    """    
    def __init__(self, Rupture: BaseRupture, Stations: Stations, 
                 GMM: GMM, SCM, jitter: float=1e-6):
        """Initializes shakemap object

        Already computes the updated between-event and within-event 
        residuals at the stations. (see ._update_residuals).

        Args:
            Rupture (BaseRupture): Openquake rupture instance

            Stations (Stations): Seismic stations with recordings to compute shakemap

            GMM (GMM): Ground-motion model

            SCM (SpatialCorrelationModel): Spatial correlation model for within-event 
                residuals.

            jitter (float, optional): For numerical stability. Defaults to 1e-6.
        """        
        self.rupture = Rupture
        self.stations = Stations
        self.gmm = GMM
        self.scm = SCM
        self.jitter = jitter
        self._update_residuals()

    def _update_residuals(self):
        """Computes the between-event and within-event residuals.

        Further explanation can be found in [1] (Section 4.1, Equation 16). These computations 
        follow the algortihm proposed in [2] and implemented in the USGS shakemap system. 

        Computes and caches:
            inv_C_SS (np.array): Inverse of the seismic stations correlation matrix.
            xiB (float): Mean between-event residual of the event.
            psiBsq (float): Remaining variance of the between-event residual.
            residuals (np.array): Within-event residuals at the seismic stations.

        References:
            [1] Bodenmann et al. (2022): Dynamic Post-Earthquake Updating of Regional Damage 
                Estimates Using Gaussian Processes. In review. doi: 10.31224/2205 
            [2] Worden et al. (2018): Spatial and Spectral Interpolation of Ground-Motion
                Intensity Measure Observations. Bulletin of the Seismological Society of America.
                doi: 10.1785/0120170201

        """
        # Get GMM predictions at seismic stations
        gmm_results = self.gmm.get_mean_and_std(self.rupture, self.stations)
        # Take unique value for phi and tau. This should be modified for GMMs which 
        # have site-specific phi or tau.
        if len(np.unique(gmm_results['phi'])) > 1:
            warnings.warn('This GMM has site-specifc phi. The mean value is used!')
        if len(np.unique(gmm_results['tau'])) > 1:
            warnings.warn('This GMM has site-specifc tau. The mean value is used!')
        self.phi = np.mean(gmm_results['phi'])
        self.tau = np.mean(gmm_results['tau'])
        # Compute correlation matrix for seismic stations
        C_SS = self.scm.get_correlation_matrix(self.stations, full_cov = True)
        # Add jitter to compute the inverse
        jitter_m = np.zeros_like(C_SS)
        np.fill_diagonal(jitter_m, self.jitter) 
        self.inv_C_SS = np.linalg.inv(C_SS + jitter_m)
        # Get recorded amplitudes
        recorded_amplitudes = self.stations.get_recordings()[:, None]
        Z = np.ones_like(recorded_amplitudes)
        # Compute total residuals
        total_residuals = recorded_amplitudes - gmm_results['mu_logIM'][:, None]
        # Compute remaining variance of between-event residual
        self.psiBsq = 1 / ( (1 / self.tau**2) + (Z.T @ (self.inv_C_SS @ Z)) / self.phi**2 )
        # Compute mean between-event residual for the recorded event
        self.xiB = self.psiBsq / self.phi**2 * Z.T @ (self.inv_C_SS @ total_residuals)
        # Compute remaining within-event residuals at the seismic stations
        self.residuals = total_residuals - self.xiB

    def predict_logIM(self, sites: Sites, conditional: bool = True, 
                      full_cov: bool = False):
        """Computes mean and covariance matrix of logarithmic im at stations.

        Args:
            sites (Sites): Sites for which to predict im

            conditional (bool, optional): Flag whether to perform computations 
                conditional on seismic station recordings or not. Defaults to True.

            full_cov (bool, optional): Flag whether to return the full covariance
                matrix or only its diagonal. Note that for a large number of sites
                (>6000) you may run into memory issues if this is true. 
                Defaults to False.

        Returns:
            mean (np.array): Logarithmic mean of im with dim (n_sites, 1).

            cov_matrix (np.array): Covariance matrix of im with dim
                (n_sites, n_sites) if full_cov = True, else: (n_sites, 1).
        """        
        gmm_results = self.gmm.get_mean_and_std(self.rupture, sites)
        C_TT = self.scm.get_correlation_matrix(sites, full_cov = full_cov)
        if conditional is False:
            mean = gmm_results['mu_logIM'][:, None]
            if full_cov:
                cov_matrix = self.tau**2 + self.phi**2 * C_TT
            else:
                cov_matrix = (self.tau**2 + self.phi**2) * np.ones_like(mean)
        else:
            C_TS = self.scm.get_correlation_matrix(sites, self.stations)
            mean = (gmm_results['mu_logIM'][:, None] + self.xiB + 
                    (C_TS @ (self.inv_C_SS @ self.residuals) ) )
            if full_cov:
                cov_matrix = ((self.phi**2 + self.psiBsq) * 
                              (C_TT - C_TS @ (self.inv_C_SS @ C_TS.T ) ) )
            else:
               cov_matrix = ((self.phi**2 + self.psiBsq) * 
                              (C_TT - np.diag(C_TS @ (self.inv_C_SS @ C_TS.T)).reshape(-1,1) ) )
        return mean, cov_matrix
    
    def sample_logIM(self, sites: Sites, nsamples: int, conditional: bool = True, 
                     seed: int = None, full_cov: bool = True):
        """Generates samples from the predicted multivariate normal of logarithmic im at sites

        Args:
            sites (Sites): Sites for which to sample im

            nsamples (int): Number of samples to generate

            conditional (bool, optional): Flag whether to perform computations 
                conditional on seismic station recordings or not. Defaults to True.

            seed (int, optional): Seed for random number generator. Defaults to None.

            full_cov (bool, optional): Flag whether to generate correlated (True) or independent 
                samples (False). Defaults to True.

        Returns:
            sim (np.array): Sampled logarithmic im with dim (n_sites x nsamples)
        """        
        rng = np.random.default_rng(seed)
        mean, cov_matrix = self.predict_logIM(sites, conditional=conditional, full_cov=full_cov)
        if full_cov:
            sim = rng.multivariate_normal(mean.squeeze(), cov_matrix, nsamples)
        else:
            sim = rng.normal(loc = mean[:,0], 
                             scale = np.sqrt(cov_matrix[:,0]), 
                             size = (nsamples, sites.n_sites) )
        return sim