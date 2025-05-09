# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:38:33 2023

@author: Luisa Lucchese

Reproject the HLS into WGS 84

inputs:
    - filename: filename of HLS file with path
    - output_raster_intermed: path to save the intermediary file

"""

def reproject_HLS(filename,output_raster_intermed):
    from osgeo import gdal
    import numpy as np
    warp_options = gdal.WarpOptions(
    dstSRS='EPSG:4326',
    resampleAlg=gdal.GRA_NearestNeighbour,
    format='GTiff',
    dstNodata=np.nan
    )
    warp = gdal.Warp(destNameOrDestDS=output_raster_intermed,srcDSOrSrcDSTab=filename,options=warp_options)
    warp = None # Closes the files
    return None