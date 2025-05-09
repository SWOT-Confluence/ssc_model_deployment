# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:56:32 2023

@author: Luisa Lucchese

Reproject MERIT DEM into UTM

inputs:
    -pointlat: latitude of in situ point
    -pointlon: longitude of in situ point
    -filename_open: MERIT DEM file to be reprojected
    -DEM: cut and reprojected DEM path
    -DEM_intermed: path to an intermediate DEM saved (different from the above)
    -distUTM: the distance, in meters, around the needed point, in which we
    will calculate shadow (larger is better but slower)
    
outputs:
    -build_epsg_name: name of the epsg in string format
    
"""

# DEM = r"D:/Luisa/start_insitu_data_wrongdata/MERIT_DEM_UTM/merit_reproj.tif"
# output_raster_warped2= r"D:/Luisa/start_insitu_data_wrongdata/MERIT_DEM_UTM/merit_reproj2.tif"

# filename_open=r"D:/MERIT_DEM_downloaded/dem_tif_n30w090/n40w075_dem.tif"

# pointlon=-73.9978
# pointlat=40.23028
# dist_pix=10 #number of pixels on each side
# distUTM=500

def reproject_to_UTM(pointlat,pointlon,filename_open,DEM,DEM_intermed,distUTM):

    import utm #need to install package
    #import osr
    from osgeo import gdal
    #conda install -c conda-forge utm

    
    (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(pointlat, pointlon)
    
    print(str(eastingpt)+ '   ' + str(northingpt))
    
    #epsg code calculation based on this https://gis.stackexchange.com/a/365589/179502
    zone = str(zone_number)
    epsg_code = 32600
    epsg_code += int(zone)
    #if south is True:
    if (pointlat<0):
        epsg_code += 100
    
    #print (epsg_code) # will be 32736
    
    #spatref = osr.SpatialReference()
    #spatref.ImportFromEPSG(epsg_code)
    #wkt_crs = spatref.ExportToWkt()
    #print (wkt_crs) #OK
    gdal.UseExceptions()
    
    build_epsg_name='EPSG:'+str(epsg_code)
    
    fileopen = gdal.Open(filename_open)
    #get location and size of the file
    #GT_input = fileopen.GetGeoTransform()
    #minx = GT_input[0] #lon
    #maxy = GT_input[3] #lat
    #maxx = minx + GT_input[1] * fileopen.RasterXSize #step1.RasterXSize is the same as size2
    #miny = maxy + GT_input[5] * fileopen.RasterYSize
    #projection=checking_outbounds.GetProjection()
    #output_bounds='('+str(pointlat-(abs(GT_input[1])*dist_pix))+', '+str(miny)+', '+str(maxx)+', ' + str(maxy)+')'
    
    #output_bounds='('+str(minx)+', '+str(miny)+', '+str(maxx)+', ' + str(maxy)+')'
    
    #code output boundaries
    #'outputBounds':output_bounds,'outputBoundsSRS':'EPSG:4326'
    output_bounds='('+str(eastingpt-distUTM)+', '+str(northingpt-distUTM)+', '+str(eastingpt+distUTM)+', ' + str(northingpt+distUTM)+')'
    
    print(output_bounds)
    
    output_raster_warped=DEM_intermed
    #output_raster_warped = 'D:\Luisa\start_insitu_data_wrongdata\downloaded\projected\warped5.tif'
    #kwargs = {'format': 'GTiff', 'dstSRS': build_epsg_name,'resampleAlg':'near','width':20,'height':20,'outputBounds':output_bounds,'outputBoundsSRS':'EPSG:4326'}
    kwargs = {'format': 'GTiff', 'dstSRS': build_epsg_name,'resampleAlg':'near'}#,'width':20,'height':20,'outputBounds':output_bounds}
    warped = gdal.Warp(output_raster_warped,fileopen,**kwargs)
    fileopen=None
    #warp is not being able to both reproject and cut, do it in parts
    
    # minX, minY, maxX, maxY
    new_bounds = [
        eastingpt-distUTM,
        northingpt+distUTM,
        eastingpt+distUTM,
        northingpt-distUTM
    ]
    
    # dest, src, options
    # http://gdal.org/python/osgeo.gdal-module.html#Translate
    warped2=gdal.Translate(DEM, output_raster_warped, projWin = new_bounds)
    warped=None
    warped2=None

    return build_epsg_name
