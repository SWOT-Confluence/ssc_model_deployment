# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:16:13 2023

@author: Luisa Lucchese

Crop one HLS image to 1km x 1km (500m radius around in situ data)

inputs:
    - filename: HLS filename with path
    - pointlon: longitude of in situ datapoint
    - pointlat: latitude of in situ datapoint
    - siteid: the id of the site where measurement happened
    - pathtocropped: path to the cropped file, where it should be saved
    - buffersize: half the x or y size the final raster should have 

"""
import os


def crop_sunmask_save(minx,miny,maxx,maxy,checking_outbounds,crop_path,output_raster_warped):
    from osgeo import gdal

    output_proj='EPSG:4326'

    new_bounds_warp = [
        minx,
        miny,
        maxx,
        maxy]
    
    geotransform = list(checking_outbounds.GetGeoTransform())
    resx_output=abs(geotransform[1])
    resy_output=abs(geotransform[5])

    warp_options = gdal.WarpOptions(
    srcSRS=output_proj,
    dstSRS=output_proj,
    xRes=resx_output,
    yRes=resy_output,
    resampleAlg=gdal.GRA_NearestNeighbour,
    outputBounds=new_bounds_warp,
    outputBoundsSRS=output_proj,
    format='GTiff'
    )

    crop_s = gdal.Warp(destNameOrDestDS=crop_path, srcDSOrSrcDSTab=output_raster_warped, options=warp_options)
    crop_s=None

    return None



def crop_HLS_save(filename,pointlon,pointlat,siteid,pathtocropped,buffersize,source,nbands,output_raster_warped):
    from osgeo import gdal
    import utm
    
    from ssc_functions.extract_info_HLS_filename import extract_info_HLS_filename
    from ssc_functions.build_HLS_filenames import build_HLS_filenames
    from ssc_functions.build_HLS_filenames_extrabands import build_HLS_filenames_extrabands
    
    # filename = 'D:/Luisa/start_insitu_data_wrongdata/downloaded/HLS.S30.T18TWK.2021250T154809.v2.0.B01.subset.tif'
    # pointlon=-73.9978
    # pointlat=40.23028
    
    # siteid='MNPCA-69-0249-00-102' #the unique index of the site, needed to save the files correctly.
    
    # pathtocropped='D:/Luisa/start_insitu_data_wrongdata/crops/'
    
    # buffersize=500 #500 m
    
    actual_filename, path_to_file, band, tile, date, time, version, source=extract_info_HLS_filename(filename)
    
    #now, reproject this into the same grid of the HLS.
    #we can use gdal warp for that
    checking_outbounds = gdal.Open(filename,gdal.GA_ReadOnly)
    #get location and size of the file
    GT_input = checking_outbounds.GetGeoTransform()
    
    
    #below, the legacy processing (useful for removing out of the bounds areas 
    #from the cropped square)
# =============================================================================
#         
#     minx_orig = GT_input[0] #lon
#     maxy_orig = GT_input[3] #lat
#     maxx_orig = minx_orig + GT_input[1] * checking_outbounds.RasterXSize #step1.RasterXSize is the same as size2
#     miny_orig = maxy_orig + GT_input[5] * checking_outbounds.RasterYSize
#     
#     #produce area around in situ to be saved
#     
#     (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(pointlat, pointlon)
#         
#     
#     minx_is=eastingpt-buffersize
#     miny_is=northingpt-buffersize
#     maxx_is=eastingpt+buffersize
#     maxy_is=northingpt+buffersize
#     
#     #initializing the new_bounds array
#     new_bounds = [
#         minx_orig,
#         maxy_orig,
#         maxx_orig,
#         miny_orig
#     ]
#     
#     print('old bounds: '+str(new_bounds))
#     
#     # testing the four borders of the HLS
#     if (minx_is > minx_orig):
#         new_bounds[0]=minx_is
#     if (maxy_is < maxy_orig):
#         new_bounds[1]=maxy_is    
#     if (maxx_is < maxx_orig):
#         new_bounds[2]=maxx_is
#     if (miny_is > miny_orig):
#         new_bounds[3]=miny_is
# =============================================================================
    
    #replaced by the simpler version (leaves NaN borders in out-of-bound areas)
    
    #produce area around in situ to be saved
    (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(pointlat, pointlon)
            
    minx_is=eastingpt-buffersize
    miny_is=northingpt-buffersize
    maxx_is=eastingpt+buffersize
    maxy_is=northingpt+buffersize
    
    #initializing the new_bounds array
    new_bounds = [
        minx_is,
        maxy_is,
        maxx_is,
        miny_is
    ]   

    # dest, src, options
    print('cropping images, new bounds: '+str(new_bounds))
    
    #run band loop
    #regular band loop
    for iband in range(1,(nbands+1)):
        try:
            filename_extract=build_HLS_filenames(date,time,pointlon,pointlat,path_to_file,iband, source)
            #using GDAL Translate
            # http://gdal.org/python/osgeo.gdal-module.html#Translate
            output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
            cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
            cropHLS=None
        except:
           print('Could not read HLS, band number: ', str(iband))
    
    #8A Band
    iband=iband+1
    try:
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'B8A')
        output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
        cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
        cropHLS=None
    except:
       print('Could not read HLS, band number: 8A')
    
    #Fmask 
    filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'Fmask')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #SAA
    filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'SAA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #SZA 
    filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'SZA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #VAA 
    filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'VAA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #VZA 
    filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'VZA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #sunmask
    iband=iband+1
    crop_path=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    crop_sunmask_save(minx_is,miny_is,maxx_is,maxy_is,checking_outbounds,crop_path,output_raster_warped)


    return None

###################################

"""
Created on Fri Mar  3 11:16:13 2023

@author: Luisa Lucchese

Crop one HLS image to 1km x 1km (500m radius around in situ data)
Specially for partial match cases

inputs:
    - filename: HLS filename with path
    - pointlon: longitude of in situ datapoint
    - pointlat: latitude of in situ datapoint
    - siteid: the id of the site where measurement happened
    - pathtocropped: path to the cropped file, where it should be saved
    - buffersize: half the x or y size the final raster should have 
    - ncrop: number of characters to cut from HLS original band name

"""


def crop_HLS_save_flagMGRS1(filename,pointlon,pointlat,siteid,pathtocropped,buffersize,ncrop,source,nbands,output_raster_warped):
    from osgeo import gdal
    import utm
    
    from ssc_functions.extract_info_HLS_filename import extract_info_HLS_filename
    from ssc_functions.build_HLS_filenames_extrabands import build_HLS_filenames_extrabands_namepres
    
    # filename = 'D:/Luisa/start_insitu_data_wrongdata/downloaded/HLS.S30.T18TWK.2021250T154809.v2.0.B01.subset.tif'
    # pointlon=-73.9978
    # pointlat=40.23028
    
    # siteid='MNPCA-69-0249-00-102' #the unique index of the site, needed to save the files correctly.
    
    # pathtocropped='D:/Luisa/start_insitu_data_wrongdata/crops/'
    
    # buffersize=500 #500 m
    
    actual_filename, path_to_file, band, tile, date, time, version, source=extract_info_HLS_filename(filename)
    
    #now, reproject this into the same grid of the HLS.
    #we can use gdal warp for that
    checking_outbounds = gdal.Open(filename,gdal.GA_ReadOnly)
    #get location and size of the file
    GT_input = checking_outbounds.GetGeoTransform()
    
    
    #below, the legacy processing (useful for removing out of the bounds areas 
    #from the cropped square)
# =============================================================================
#         
#     minx_orig = GT_input[0] #lon
#     maxy_orig = GT_input[3] #lat
#     maxx_orig = minx_orig + GT_input[1] * checking_outbounds.RasterXSize #step1.RasterXSize is the same as size2
#     miny_orig = maxy_orig + GT_input[5] * checking_outbounds.RasterYSize
#     
#     #produce area around in situ to be saved
#     
#     (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(pointlat, pointlon)
#         
#     
#     minx_is=eastingpt-buffersize
#     miny_is=northingpt-buffersize
#     maxx_is=eastingpt+buffersize
#     maxy_is=northingpt+buffersize
#     
#     #initializing the new_bounds array
#     new_bounds = [
#         minx_orig,
#         maxy_orig,
#         maxx_orig,
#         miny_orig
#     ]
#     
#     print('old bounds: '+str(new_bounds))
#     
#     # testing the four borders of the HLS
#     if (minx_is > minx_orig):
#         new_bounds[0]=minx_is
#     if (maxy_is < maxy_orig):
#         new_bounds[1]=maxy_is    
#     if (maxx_is < maxx_orig):
#         new_bounds[2]=maxx_is
#     if (miny_is > miny_orig):
#         new_bounds[3]=miny_is
# =============================================================================
    
    #replaced by the simpler version (leaves NaN borders in out-of-bound areas)
    
    #produce area around in situ to be saved
    (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(pointlat, pointlon)
            
    minx_is=eastingpt-buffersize
    miny_is=northingpt-buffersize
    maxx_is=eastingpt+buffersize
    maxy_is=northingpt+buffersize
    
    #initializing the new_bounds array
    new_bounds = [
        minx_is,
        maxy_is,
        maxx_is,
        miny_is
    ]   

    # dest, src, options
    print('new bounds: '+str(new_bounds))
    
    #run band loop
    #regular band loop
    for iband in range(1,(nbands+1)):
        try:
            bandnum_str='B'+str(iband).zfill(2)
            filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,bandnum_str,ncrop) #build_HLS_filenames(date,time,pointlon,pointlat,path_to_file,iband)
            #using GDAL Translate
            # http://gdal.org/python/osgeo.gdal-module.html#Translate
            output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
            cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
            cropHLS=None
        except:
           print('Could not read HLS, band number: ', str(iband))
    
    #8A Band
    iband=iband+1
    try:
        filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'B8A',ncrop)
        #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'B8A')
        output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
        cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
        cropHLS=None
    except:
       print('Could not read HLS, band number: 8A')
    
    #Fmask 
    filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'Fmask',ncrop)
    #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'Fmask')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #SAA
    filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'SAA',ncrop)
    #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'SAA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #SZA 
    filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'SZA',ncrop)
    #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'SZA')
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #VAA 
    #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'VAA')
    filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'VAA',ncrop)
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #VZA 
    #filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,'VZA')
    filename_extract=fmask=build_HLS_filenames_extrabands_namepres(filename,'VZA',ncrop)
    iband=iband+1
    output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    cropHLS=None
    
    #sunmask
    iband=iband+1
    crop_path=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    crop_sunmask_save(minx_is,miny_is,maxx_is,maxy_is,checking_outbounds,crop_path,output_raster_warped)

    return None    



