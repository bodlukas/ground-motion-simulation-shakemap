import numpy as np
import pandas as pd
from openquake.hazardlib.geo.surface.planar import PlanarSurface
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.surface.multi import MultiSurface
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.geodetic import point_at, geodetic_distance

#---
# Finite Fault: Version 4 -> https://earthquake.usgs.gov/earthquakes/eventpage/us6000jllz/finite-fault 
# Modelled as three segments, each with a planar surface. 
# Read from map top_left point and compute the other corner points with given rake and distance.
# Validated that the resulting dip is the same.
#---

finite_fault_geom_v4 = dict()
finite_fault_geom_v4['top_left'] = [[36.918, 37.084, 0], [36.825, 37.370, 0], [36.053, 36.073, 0]]
finite_fault_geom_v4['length_1'] = [54.975, 189.859, 160.133]  
finite_fault_geom_v4['length_2'] = [3.495, 3.491, 10.503]
finite_fault_geom_v4['strike'] = [28, 60, 25]
finite_fault_geom_v4['depth'] = [40, 40, 40]

def get_finite_fault():
    rup_geom = finite_fault_geom_v4
    num_segments = len(rup_geom['strike'])
    coors = []
    for i in range(num_segments):
        top_left = rup_geom['top_left'][i]
        top_right = point_at(top_left[0], top_left[1], rup_geom['strike'][i], rup_geom['length_1'][i])
        bottom_left = point_at(top_left[0], top_left[1], rup_geom['strike'][i]+90, rup_geom['length_2'][i])
        bottom_right = point_at(bottom_left[0], bottom_left[1], rup_geom['strike'][i], rup_geom['length_1'][i])
        c_temp = [[top_left[0], top_left[1], 0],
                [top_right[0], top_right[1], 0],
                [bottom_right[0], bottom_right[1], rup_geom['depth'][i]],
                [bottom_left[0], bottom_left[1], rup_geom['depth'][i]],
                [top_left[0], top_left[1], 0]]
        
        coors.append(c_temp)
    rup_geom['coordinates'] = [coors]
    return rup_geom

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

def get_extent_size(ax, rounding=0):
    lonsmin, lonsmax = ax.get_xlim()
    latsmin, latsmax = ax.get_ylim()
    size_x = np.mean([geodetic_distance(lonsmin, latsmin, lonsmax, latsmin),
                    geodetic_distance(lonsmin, latsmax, lonsmax, latsmax)])
    size_y = np.mean([geodetic_distance(lonsmin, latsmin, lonsmin, latsmax),
                    geodetic_distance(lonsmax, latsmin, lonsmax, latsmax)])
    return (np.round(size_x, rounding), np.round(size_y, rounding))