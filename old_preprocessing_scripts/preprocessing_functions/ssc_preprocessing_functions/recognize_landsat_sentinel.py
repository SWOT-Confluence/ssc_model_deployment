# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:21:15 2023

@author: Luisa Lucchese
"""
def recognize_landsat_sentinel(actual_filename):
    if actual_filename[0:4]=='HLS.':
        source=actual_filename[4]
        print('It is an HLS file of source: ', source)
    else:
        print('Attention: it is not an HLS file.')
        source='N'
    return source
