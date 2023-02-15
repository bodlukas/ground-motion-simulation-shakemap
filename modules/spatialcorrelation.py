import numpy as np
import pandas as pd
from openquake.hazardlib import imt
from openquake.hazardlib.geo.geodetic import geodetic_distance
import math

class SpatialCorrelationModel(object):
    """

    Base class for spatial correlation models of within-event residuals 
    of ground-motion intensity measures

    :param im:
        string that indicates intensity measure (i.e. 'PGA' or 'SA(1.0)')
        should be compatible with openquake intensity measure names

    """
    def __init__(self, im):
        self.im = imt.from_string(im)

    def get_euclidean_distance_matrix(self, sites1, sites2=None, full_cov=True):
        ''' 
        Computes the Euclidean distance between each site in sites1 and each
        site in site2

        Parameters
        ----------
        sites1 : Sites object (see shakemap.py)
            n sites
        sites2 : Sites object (see shakemap.py)
            m sites
            if None: compute distances within sites 1
        sites2 : Sites object (see shakemap.py)
            m sites
        full_cov : bool
            If False computes only the diagonal of the distance matrix (zeros)

        Returns
        -------
        distances : np.array
            if sites2 is None: distance matrix of sites1 with themselves (dim n x n)
            else: distance matrix between sites1 and sites2 (dim n x m)
        '''
        lons1 = sites1.mesh.lons.reshape(-1,1); lats1 = sites1.mesh.lats.reshape(-1,1)
        if sites2 is None:
            if full_cov:
                distances = geodetic_distance(lons1, lats1, lons1.T, lats1.T)
            else:
                distances = np.zeros_like(lons1)
        else:
            lons2 = sites2.mesh.lons.reshape(1,-1); lats2 = sites2.mesh.lats.reshape(1,-1)
            distances = geodetic_distance(lons1, lats1, lons2, lats2)
        return distances
    
class HeresiMiranda2019(SpatialCorrelationModel):
    '''
    
    Implements model: 
        Heresi P. and Miranda E. (2019): "Uncertainty in intraevent spatial 
        correlation of elastic pseudo‑acceleration spectral ordinates"
        Bulletin of Earthquake Engineering, doi: 10.1007/s10518-018-0506-6

    Additional inputs:
        - mode: 'median' (default) or 'mean'

    '''
    def __init__(self, im, mode='median'):
        super().__init__(im)
        self.T = self.im.period
        self.mode = mode

    def get_correlation_matrix(self, sites1, sites2=None, full_cov=True):
        dist_mat = self.get_euclidean_distance_matrix(sites1, sites2, full_cov)
        beta = self._get_parameter_beta()
        return np.exp( - np.power(dist_mat / beta, 0.55) )
    
    def _get_parameter_beta(self):
        if self.T < 1.37:
            Med_b = 4.231 * self.T * self.T - 5.180 * self.T + 13.392
        else:
            Med_b = 0.140 * self.T * self.T - 2.249 * self.T + 17.050

        if self.mode == 'median':
            return Med_b
        elif self.mode == 'mean':
            Std_b = (4.63e-3 * self.T*self.T + 0.028 * self.T + 0.713)
            return np.exp( np.log(Med_b) + 0.5 * Std_b)

class EspositoIervolino2012esm(SpatialCorrelationModel):
    '''
    Implements model proposed in:

    - for PGA: 
        Esposito S. and Iervolino I. (2011): "PGA and PGV Spatial Correlation Models 
        Based on European Multievent Datasets"
        Bulletin of the Seismological Society of America, doi: 10.1785/0120110117

    - for SA(T): 
        Esposito S. and Iervolino I. (2012): "Spatial Correlation of Spectral 
        Acceleration in European Datas"
        Bulletin of the Seismological Society of America, doi: 10.1785/0120120068
    
    Considers only parameters derived from the European ground-motion dataset !

    '''

    def __init__(self, im):
        super().__init__(im)
        self.T = self.im.period

    def get_correlation_matrix(self, sites1, sites2=None, full_cov=True):
        dist_mat = self.get_euclidean_distance_matrix(sites1, sites2, full_cov)
        range = self._get_parameter_range()
        return np.exp(-3 * dist_mat / range)
    
    def _get_parameter_range(self):
        if self.T == 0: range = 13.5
        else: range = 11.7 + 12.7 * self.T
        return range

# Parameters for Model of BodenmannEtAl
params_BodenmannEtAl = {0.01: {'LE': 16.4, 'gammaE': 0.36, 'LA': 24.9, 'LS': 171.2, 'w': 0.84},
                        0.03: {'LE': 16.9, 'gammaE': 0.36, 'LA': 25.6, 'LS': 185.6, 'w': 0.84},
                        0.06: {'LE': 16.6, 'gammaE': 0.35, 'LA': 24.4, 'LS': 190.2, 'w': 0.84},
                        0.1: {'LE': 16.3, 'gammaE': 0.34, 'LA': 23.3, 'LS': 189.8, 'w': 0.88},
                        0.3: {'LE': 15.1, 'gammaE': 0.34, 'LA': 26.1, 'LS': 199.9, 'w': 0.85},
                        0.6: {'LE': 25.6, 'gammaE': 0.37, 'LA': 24.2, 'LS': 222.8, 'w': 0.73},
                        1.0: {'LE': 29.8, 'gammaE': 0.41, 'LA': 20.5, 'LS': 169.2, 'w': 0.7},
                        3.0: {'LE': 42.1, 'gammaE': 0.46, 'LA': 18.5, 'LS': 358.0, 'w': 0.5},
                        6.0: {'LE': 70.2, 'gammaE': 0.49, 'LA': 17.3, 'LS': 372.2, 'w': 0.54}}


class BodenmannEtAl2022(SpatialCorrelationModel):
    '''
     Implements model: 
        Bodenmann L., Baker J.W. and Stojadinovic B. (2022): "Accounting for path and site effects in 
        spatial ground-motion correlation models using Bayesian inference"
        Natural Hazards and Earth System Sciences (in review), doi: 10.5194/nhess-2022-267

    Additional inputs:
        - rupture: OpenQuake BaseRupture object

    Note: For PGA, we take the parameters obtained for SA(T=0.01s).
    '''

    def __init__(self, im: str, rupture):
        super().__init__(im)
        self.T = self.im.period
        self.rupture = rupture

    def compute_epicentral_azimuth(self, lons, lats):
        """
        Calculate the azimuths of a collection of points with respect to the epicenter.
        """
        lon, lat = self.rupture.hypocenter.longitude, self.rupture.hypocenter.latitude
        lon, lat = math.radians(lon), math.radians(lat)
        lons, lats = np.radians(lons), np.radians(lats)
        cos_lats = np.cos(lats)
        true_course = np.arctan2(
            np.sin(lon - lons) * cos_lats,
            math.cos(lat) * np.sin(lats) -
            math.sin(lat) * cos_lats * np.cos(lon - lons))
        return - np.degrees(true_course)


    def get_angular_distance_matrix(self, sites1, sites2=None, full_cov=True):
        '''
        Computes matrix with difference in epicentral azimuth values of sites:

            if sites2 is None: distance matrix of sites1 with themselves
            else: distance matrix between sites1 and sites2

            if full_cov is True: Returns full distance matrix
            else: Returns only the diagonal (zeros)
        '''
        azimuths1 = np.radians(
            self.compute_epicentral_azimuth(sites1.mesh.lons, sites1.mesh.lats))
        azimuths1 = azimuths1.reshape(-1, 1)

        if sites2 is None:
            if full_cov:
                cos_angle = np.cos( np.abs(azimuths1 - azimuths1.T) )
                distances =  np.arccos(np.clip(cos_angle, -1, 1))
            else:
                distances =  np.zeros_like(azimuths1)      
        else:
            azimuths2 = np.radians(
                self.compute_epicentral_azimuth(sites2.mesh.lons, sites2.mesh.lats))
            azimuths2 = azimuths2.reshape(1, -1)
            cos_angle = np.cos( np.abs(azimuths1 - azimuths2) )
            distances =  np.arccos(np.clip(cos_angle, -1, 1))
        return distances * 180/np.pi
    
    def get_soil_dissimilarity_matrix(self, sites1, sites2=None, full_cov=True):
        '''
        Computes the absolute difference in vs30 values between sites:

            if sites2 is None: distance matrix of all sites in sites1
            else: distance matrix between sites1 and sites2 

            if full_cov is True: Returns full distance matrix
            else: Returns only the diagonal (zeros)
        '''
        vs301 = sites1.vs30.reshape(-1,1)

        if sites2 is None:
            if full_cov:
                distances = np.abs(vs301 - vs301.T)
            else:
                distances =  np.zeros_like(vs301)      
        else:
            vs302 = sites2.vs30.reshape(1, -1)
            distances = np.abs(vs301 - vs302)
        return distances

    def get_correlation_matrix(self, sites1, sites2=None, full_cov=True):
        dist_mat_E = self.get_euclidean_distance_matrix(sites1, sites2, full_cov)
        dist_mat_A = self.get_angular_distance_matrix(sites1, sites2, full_cov)
        dist_mat_S = self.get_soil_dissimilarity_matrix(sites1, sites2, full_cov)
        params = self._get_parameters()
        rho_E = np.exp( - np.power(dist_mat_E / params['LE'], params['gammaE']) )
        rho_A = (1 + dist_mat_A/params['LA']) * np.power(1 - dist_mat_A/180, 180/params['LA'])
        rho_S = np.exp(- dist_mat_S / params['LS'])
        return rho_E * (params['w'] * rho_A + (1-params['w']) * rho_S)
    
    def _get_parameters(self):
        if self.T == 0:
            return params_BodenmannEtAl[0.01]
        else:
            return params_BodenmannEtAl[self.T]