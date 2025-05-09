# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:18:54 2023

@author: Luisa Lucchese

This is the script to read values from the HLS rasters,
apply the masks and return the readings. 

inputs:
    - filename: HLS filename
    - pointlon: longitude of the point
    - pointlat: latitude of the point
    - sunmask_warp: sunmask, warped to the exact same extent as the HLS
    - output_raster_intermed: path of the HLS, projected and ready to be used
    - mask_Fmask: the Fmask, to be used in masking
outputs:
    - attr_point: minimum, average, maximum, and standard deviation values for 
    the band.
"""

#created an environment for this processing using Anaconda prompt
#conda create --name procHLS
#conda install gdal ATTENTION CURRENT VERSION GDAL IS BROKEN
#conda install -c conda-forge gdal NO THIS VERSION IS BROKEN
#conda install -c conda-forge libgdal NO THIS VERSION IS BROKEN
# check info at https://github.com/conda-forge/gdal-feedstock/issues/541
#conda install gdal=3.0

#installing pyproj by this answer 
#https://stackoverflow.com/questions/343865/how-to-convert-from-utm-to-latlng-in-python-or-javascript
#conda install -c conda-forge pyproj 
#NO, LETS REPROJECT THE FILES WITH GDAL FIRST

#activate environment before running this script
# 


def extract_value(filename,pointlon,pointlat,sunmask_warp,output_raster_intermed,mask_Fmask, dst_filename_wos,dst_filename_sun):

    #
    import numpy as np
    import osr
    from osgeo import gdal
    import math
    #from pyproj import Transformer, CRS

    #buffer size (in km) approximate
    bufsize=.3

    #point from which it comes -- pointlon,pointlat

    # reprojecting using warp 
    # as per https://gis.stackexchange.com/questions/233589/re-project-raster-in-python-using-gdal

    print('Extracting value from', output_raster_intermed)
    step1 = gdal.Open(output_raster_intermed, gdal.GA_ReadOnly)
    
    #get location and size of the file
    GT_input = step1.GetGeoTransform()
    minx = GT_input[0] #lon
    maxy = GT_input[3] #lat
    maxx = minx + GT_input[1] * step1.RasterXSize #step1.RasterXSize is the same as size2
    miny = maxy + GT_input[5] * step1.RasterYSize
    #projection=step1.GetProjection()

    step2 = step1.GetRasterBand(1)

    img_as_array = step2.ReadAsArray()

    size1,size2=img_as_array.shape

    output=np.zeros(shape=(size1,size2))

    for i in range(0,size1):
        for j in range(0,size2):
            if (img_as_array[i,j]==-9999):
                output[i,j]=np.nan#img_as_array[i,j] ** 1.0 
            else:
                output[i,j]=img_as_array[i,j]
    
    print('extract att1',np.nansum(output)) #check if the array has valid numbers

    #doing the buffer
    output2=np.zeros(shape=(size1,size2))
    # print('output2 shape', size1, size2)
    output3=np.zeros(shape=(size1,size2)) #initialize variable
    # https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    #Length in km of 1° of latitude = always 111.32 km
    # 1D = 111.32 , xD= 0.2 --> 0.2 = 111.32x --> x=0.001796622349982034
    #Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360
    # 0.2km * 360 / 40075 km =cos(latitude)
    #  (0.2 km * 360/40075)/cos(latitude)=delta_lat

    #buffer size y (in degrees)
    bufsize_y=bufsize/111.32
    #buffer size x (in degrees)
    pointlat_rad=(pointlat/180)*math.pi
    bufsize_x=(bufsize*360/40075)/np.cos(pointlat_rad)

    #ellipse equation
    #x2/a2 + y2/b2 = 1
    #x2/bufsize_x2 + y2/bufsize_y2 =1

    bufsizexquad=bufsize_x**2 #to calculate only once
    bufsizeyquad=bufsize_y**2 #to calculate only once

    #create area around point ("buffer")
    #buffer_counter=0 #count the number of points got inside the buffer
    for i in range(0,(size1)): #+1

        for j in range(0,(size2)):
            # print('sz1',size1)
            # print('sz2', size2) #+1
            # raise ValueError
            if (GT_input[1]>0): #necessary to keep the order of the direction
                coordlocal_x=minx + j * GT_input[1]
            else:
                coordlocal_x=maxx + j * GT_input[1]    
            coordloctransf_x=coordlocal_x - pointlon #transformation of referential
            if (GT_input[5]>0): #same for y, necessary to keep the order of the direction
                coordlocal_y=miny + i * GT_input[5]
            else:
                coordlocal_y=maxy + i * GT_input[5] 
                #
            coordloctransf_y= coordlocal_y - pointlat #transformation of referential
            
            if (coordloctransf_x**2/bufsizexquad + coordloctransf_y**2/bufsizeyquad < 1): #inside the ellipsoid
                #buffer_counter=buffer_counter+1
                # print('got point')
                # print('i', i)
                # print('j', j)
                # print('single point out', output[i,j])
                # print('fmask shape',mask_Fmask)
                output2[i,j]=output[i,j]*mask_Fmask[i,j] #uncomment this last part when good sunmask is availabe
                # print('fmask point', output2[i,j])
                # print('sunmask shape', sunmask_warp)
                output3[i,j]=output2[i,j]*sunmask_warp[i,j] # this is the output with the sunmask
                # print('sunmask point', output2[i,j])
                #print('sunmask_warp='+str(sunmask_warp[i,j]))
            else:
                output2[i,j]=np.nan
                output3[i,j]=np.nan
    #print(output2)
    print('output2', np.nansum((output2))) #check if the output has changed #it has
    print('output3', np.nansum((output3))) #
    
    buffer_counter_wos=np.count_nonzero(~np.isnan(output2))
    buffer_counter_sun=np.count_nonzero(~np.isnan(output3))
    
    print('number of points inside the buffer without sunmask: '+str(buffer_counter_wos)) 
    print('number of points inside the buffer with sunmask: '+str(buffer_counter_sun)) 
    #try to save this as a raster somehow
    #trying to use gdal itself for it and avoid rasterio completely
    #basing myself on this solution
    #https://drr.ikcest.org/tutorial/k8024
    #and on the GDAL documentation


    ## saving output WITHOUT sunmask
    driver = gdal.GetDriverByName( 'GTiff' )
    # /media/travis/work/repos/hls_full/ssc/output/intermediate
    # dst_filename = '/media/travis/work/repos/hls_full/HLS_preproc/start_insitu_data_wrongdata/downloaded/projected/intermed25.tif'
    dst_ds=driver.Create(dst_filename_wos,size2,size1,1,gdal.GDT_Float64)
    dst_ds.SetGeoTransform(GT_input)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'WGS84' )
    dst_ds.GetRasterBand(1).WriteArray( output2 )

    attr_point_min=np.nanmin(output2)
    attr_point_mean=np.nanmean(output2)
    attr_point_max=np.nanmax(output2)
    attr_point_std=np.nanstd(output2)
    attr_point_median=np.nanmedian(output2)
    
    attr_point_wos=[attr_point_min,attr_point_mean,attr_point_max,attr_point_std,attr_point_median,buffer_counter_wos]

    ## saving output WITH sunmask
    driver = gdal.GetDriverByName( 'GTiff' )
    # /media/travis/work/repos/hls_full/ssc/output/intermediate
    # dst_filename = '/media/travis/work/repos/hls_full/HLS_preproc/start_insitu_data_wrongdata/downloaded/projected/intermed25.tif'
    dst_ds=driver.Create(dst_filename_sun,size2,size1,1,gdal.GDT_Float64)
    dst_ds.SetGeoTransform(GT_input)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'WGS84' )
    dst_ds.GetRasterBand(1).WriteArray( output3 )

    attr_point_min=np.nanmin(output3)
    attr_point_mean=np.nanmean(output3)
    attr_point_max=np.nanmax(output3)
    attr_point_std=np.nanstd(output3)
    attr_point_median=np.nanmedian(output3)
    
    attr_point_sun=[attr_point_min,attr_point_mean,attr_point_max,attr_point_std,attr_point_median,buffer_counter_sun]


    return np.transpose(attr_point_wos), np.transpose(attr_point_sun)

