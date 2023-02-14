import numpy as np
import pandas as pd
from openquake.hazardlib import imt
from openquake.hazardlib.geo.geodetic import geodetic_distance

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

    def get_euclidean_distance_matrix(self, mesh1, mesh2=None, full_cov=True):
        '''
        Computes distance matrix:

            if mesh2 is None: distance matrix of all sites in mesh1
            else: distance matrix between sites in mesh1 and sites in mesh

            if full_cov is True: Returns full distance matrix
            else: Returns only the diagonal (zeros)
        '''
        lons1 = mesh1.lons.reshape(-1,1); lats1 = mesh1.lats.reshape(-1,1)
        if mesh2 is None:
            if full_cov:
                distances = geodetic_distance(lons1, lats1, lons1.T, lats1.T)
            else:
                distances = np.zeros_like(lons1)
        else:
            lons2 = mesh2.lons.reshape(1,-1); lats2 = mesh2.lats.reshape(1,-1)
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

    def get_correlation_matrix(self, mesh1, mesh2=None, full_cov=True):
        dist_mat = self.get_euclidean_distance_matrix(mesh1, mesh2, full_cov)
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

    def get_correlation_matrix(self, mesh1, mesh2=None, full_cov=True):
        dist_mat = self.get_euclidean_distance_matrix(mesh1, mesh2, full_cov)
        range = self._get_parameter_range()
        return np.exp(-3 * dist_mat / range)
    
    def _get_parameter_range(self):
        if self.T == 0: range = 13.5
        else: range = 11.7 + 12.7 * self.T
        return range