# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:14:03 2023

@author: Luisa Lucchese

reprojects the sunmask 

inputs:
    - dst_filename: the un-warped version of the sunmask (with the path to it)
    - pathintermediate: where the intermediate file is saved
    - output_raster_intermed: the intermediate projected HLS path
    - epsg_shadow: the epsg (projection) of the shadow
outputs:
    - sunamsk_as_array: the sunmask, warped to the limits and size of the HLS    
    
"""

def reproject_sunmask(dst_filename,pathintermediate,output_raster_intermed,epsg_shadow,source): #,zoneUTM,letterUTM): #for now accepting only WGS84 entries
    from osgeo import gdal,osr
    import re
    #import utm
    #now, reproject this into the same grid of the HLS.
    #we can use gdal warp for that
    print(output_raster_intermed)
    checking_outbounds = gdal.Open(output_raster_intermed,gdal.GA_ReadOnly)
    #get location and size of the file
    GT_input = checking_outbounds.GetGeoTransform()
    print(GT_input)
    minx = GT_input[0] #lon
    maxy = GT_input[3] #lat
    maxx = minx + GT_input[1] * checking_outbounds.RasterXSize #step1.RasterXSize is the same as size2
    miny = maxy + GT_input[5] * checking_outbounds.RasterYSize
    projection_use_bounds=checking_outbounds.GetProjection()
    
    # is the minimum really minimum and the maximum really maximum?
    if minx > maxx:
        savemin=maxx
        maxx=minx
        minx=savemin
    if miny > maxy:
        savemin=maxy
        maxy=miny
        miny=savemin
    
    # (latmin,lonmin)=utm.to_latlon(minx, miny, zoneUTM, letterUTM, strict=False) #only if source is UTM
    # (latmax,lonmax)=utm.to_latlon(maxx, maxy, zoneUTM, letterUTM, strict=False) #only if source is UTM
    
    input_raster = gdal.Open(dst_filename)
    output_raster_warped=dst_filename[0:-4]+'_shadowproj1.tif'#
    kwargs = {'format': 'GTiff', 'srcSRS':epsg_shadow, 'dstSRS': 'EPSG:4326'}#
    sunmask_warp = gdal.Warp(output_raster_warped,input_raster,**kwargs)
    
    # below only for UTM projected
    # new_bounds = [
    #     lonmin,
    #     latmax,
    #     lonmax,
    #     latmin
    # ]
    
    # upper_left_x 
    # upper_left_y 
    # lower_right_x 
    # lower_right_y 
    new_bounds = [
        minx,
        maxy,
        maxx,
        miny
    ]
    # dest, src, options
    print('newbounds: '+str(new_bounds)+' projection: '+str(projection_use_bounds), ' old projection epsg_shadow: '+str(epsg_shadow))
    # http://gdal.org/python/osgeo.gdal-module.html#Translate
    output_raster_warped2=dst_filename[0:-4]+'_shadowproj2.tif'
    sunmask_warp_cut=gdal.Translate(output_raster_warped2, output_raster_warped, projWin = new_bounds, projWinSRS = projection_use_bounds)
    sunmask_warp_cut=None

    output_raster_warped3=dst_filename[0:-4]+'_shadowproj3.tif'

    # 
    outbounds_input = gdal.Open(output_raster_warped2,gdal.GA_ReadOnly)
    res_input_x=outbounds_input.RasterXSize 
    res_input_y=outbounds_input.RasterYSize
    res_output_x=checking_outbounds.RasterXSize 
    res_output_y=checking_outbounds.RasterYSize

    # Using GDAL Warp
    # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Warp

    geotransform = list(checking_outbounds.GetGeoTransform())
    resx_output=abs(geotransform[1])
    resy_output=abs(geotransform[5])

    input_proj_step1 = osr.SpatialReference(wkt=outbounds_input.GetProjection())
    input_proj='EPSG:'+str(input_proj_step1.GetAttrValue('AUTHORITY',1))
    #output_proj_step1 = osr.SpatialReference(wkt=checking_outbounds.GetProjection())
    #output_proj='EPSG:'+str(output_proj_step1.GetAttrValue('AUTHORITY',1))
    output_proj='EPSG:4326'
    new_bounds_warp = [
        minx,
        miny,
        maxx,
        maxy]
    warp_options = gdal.WarpOptions(
    srcSRS=input_proj,
    dstSRS=output_proj,
    xRes=resx_output,
    yRes=resy_output,
    resampleAlg=gdal.GRA_NearestNeighbour,
    outputBounds=new_bounds_warp,
    outputBoundsSRS=output_proj,
    format='GTiff'
    )
    output_ds = gdal.Warp(destNameOrDestDS=output_raster_warped3, srcDSOrSrcDSTab=output_raster_warped2, options=warp_options)
    # elif source == 'L':
    #     try:
    #         output_proj_step1 = osr.SpatialReference(wkt=checking_outbounds.GetProjection())
    #         output_proj_step2 = str(output_proj_step1.GetAttrValue('PROJCS',0))
    #         output_proj_step3 = ''.join(filter(str.isalnum, output_proj_step2))
    #         print(output_proj_step3)
    #         output_proj_step4=re.findall(r'\d\d', output_proj_step3)
    #         if output_proj_step4[0]== '84':
    #             utmzone=output_proj_step4[1]
    #         else:
    #             utmzone=output_proj_step4[0]
    #         output_proj_step5=output_proj_step3.replace('WGS', '')
    #         output_proj_step6=output_proj_step5.replace('ZONE', '')
    #         if 'North' in output_proj_step6:
    #             south = False
    #         elif 'north' in output_proj_step6:
    #             south = False
    #         elif 'N' in output_proj_step6:
    #             south = False
    #         else:    
    #             south = True
    #         epsg_code = 32600
    #         epsg_code += int(utmzone)
    #         if south is True:
    #             epsg_code += 100
    #         print ('EPSG: '+str(epsg_code)) # 
    #         output_proj='EPSG:'+str(epsg_code)
    #         new_bounds_warp = [
    #         minx,
    #         miny,
    #         maxx,
    #         maxy]
    #         warp_options = gdal.WarpOptions(
    #         srcSRS=input_proj,
    #         dstSRS=output_proj,
    #         xRes=resx_output,
    #         yRes=resy_output,
    #         resampleAlg=gdal.GRA_NearestNeighbour,
    #         outputBounds=new_bounds_warp,
    #         outputBoundsSRS=output_proj,
    #         format='GTiff'
    #         )
    #         output_ds = gdal.Warp(destNameOrDestDS=output_raster_warped3, srcDSOrSrcDSTab=output_raster_warped2, options=warp_options)
    #         print('Option 2 worked yay')
    #     except:
    #         output_proj_step1 = osr.SpatialReference(wkt=checking_outbounds.GetProjection())
    #         output_proj_step2 = str(output_proj_step1.GetAttrValue('PROJCS',0))
    #         utmzone_step1=re.findall(r'\d+[a-zA-Z]', output_proj_step2)
    #         utmzone=re.findall(r'\d+', utmzone_step1[0])
    #         NorS=re.findall(r'[a-zA-Z]', utmzone_step1[0])
    #         if NorS== 'N' or 'n':
    #             south = False
    #         else:
    #             south = True
    #         epsg_code = 32600
    #         epsg_code += int(utmzone[0])
    #         if south is True:
    #             epsg_code += 100
    #         #print (epsg_code) # 
    #         output_proj='EPSG:'+str(epsg_code)
    #         new_bounds_warp = [
    #         minx,
    #         miny,
    #         maxx,
    #         maxy]
    #         warp_options = gdal.WarpOptions(
    #         srcSRS=input_proj,
    #         dstSRS=output_proj,
    #         xRes=resx_output,
    #         yRes=resy_output,
    #         resampleAlg=gdal.GRA_NearestNeighbour,
    #         outputBounds=new_bounds_warp,
    #         outputBoundsSRS=output_proj,
    #         format='GTiff'
    #         )
    #         output_ds = gdal.Warp(destNameOrDestDS=output_raster_warped3, srcDSOrSrcDSTab=output_raster_warped2, options=warp_options)
    #output_proj_step1 = osr.SpatialReference(wkt=checking_outbounds.GetProjection())
    #output_proj='EPSG:'+str(output_proj_step1.GetAttrValue('AUTHORITY',1))


    
    """     # Set transformation options
    warp_options = gdal.WarpOptions(
        srcSRS=input_proj,
        dstSRS=output_proj,
        xRes=resx_output,
        yRes=resy_output,
        resampleAlg=gdal.GRA_NearestNeighbour,
        outputBounds=new_bounds_warp,
        outputBoundsSRS=output_proj,
        format='GTiff'
    )

    # Perform the transformation
    output_ds = gdal.Warp(destNameOrDestDS=output_raster_warped3, srcDSOrSrcDSTab=output_raster_warped2, options=warp_options) """

    # Close datasets
    output_ds=None
    
    #adapting to the new code part
    sunmask_warp_cut2 = gdal.Open(output_raster_warped3,gdal.GA_ReadOnly)
    sunmask_warp_band = sunmask_warp_cut2.GetRasterBand(1)
    sunmask_as_array = sunmask_warp_band.ReadAsArray()
    
    #closing datasets
    checking_outbounds=None #closes the dataset
    sunmask_warp=None
    warped2=None
    sunmask_warp_cut=None
    sunmask_warp_cut2=None
    outbounds_input=None
    
    return sunmask_as_array,output_raster_warped

