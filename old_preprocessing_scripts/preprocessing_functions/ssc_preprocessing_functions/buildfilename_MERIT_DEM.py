# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:12:19 2023

@author: Luisa Lucchese

Find out which MERIT DEM file to open, based on the geographic coordinates of
a point.
inputs:
    - plon: longitude of the point
    - plat: latitude of the point
    - folder_initial: where are the DEM files (folder of folders)
    
outputs:
    - filename_w_path: filename and path to be opened

"""
import os

def buildfilename_MERIT_DEM(plon,plat,folder_initial):
    import math
    
    #plon=-73.9978
    #plat=40.23028
    
    #plon=73.9978
    #plat=40.23028
    
    #plon=78.9978
    #plat=-38.23028
    
    #plon=-68.9978 #verified for all four combinations
    #plat=-38.23028
    
    #folder_initial='D:/MERIT_DEM_downloaded/' #where are the DEM files
    
    plonabs=abs(plon)
    platabs=abs(plat)
    
    folder_prefix='dem_tif_'
    
    file_suffix='_dem.tif'
    
    #equator, is it below of above?
    
    if (plat>0):
        #above equator
        file_north='n'
        if (plat>30):
            if (plat>60):
                #between 60N and 90N
                folder_northing='n60'
            else:
                #between 30N and 60N
                folder_northing='n30'
            
        else:
            #between 0 and 30N
            folder_northing='n00'
        
    else:
        #below equator
        file_north='s'
        if(plat<-30):
            #between 30S and 60S
            folder_northing='s60'
        else:
            #between 0 and 30S
            folder_northing='s30'
    
    #hemisphere
    
    if (plon>0):
        #eastern hemisphere
        file_hemi='e'
        if (plon>60):
            if(plon>120):
                if (plon>150):
                    #between 150 and 180
                    folder_easting='e150'
                else:
                    #between 120 and 150
                    folder_easting='e120'
            else:
                if (plon>90):
                    #between 90 and 120
                    folder_easting='e090'
                else:
                    #between 60 and 90
                    folder_easting='e060'
        else:
            if (plon>30):
                #between 30 and 60
                folder_easting='e030'
            else:
                #between 0 and 30
                folder_easting='e000'
        
    else:
        #western hemisphere
        file_hemi='w'
        if (plonabs>60):
            if(plonabs>120):
                if (plonabs>150):
                    #between 150 and 180
                    folder_easting='w180'
                else:
                    #between 120 and 150
                    folder_easting='w150'
            else:
                if (plonabs>90):
                    #between 90 and 120
                    folder_easting='w120'
                else:
                    #between 60 and 90
                    folder_easting='w090'
        else:
            if (plonabs>30):
                #between 30 and 60
                folder_easting='w060'
            else:
                #between 0 and 30
                folder_easting='w030'
    
    
    foldername=folder_prefix+folder_northing+folder_easting+'/'
    
    #specific filename
    
    def flooroffive(x, base=5):
        return base * math.floor(x/base)
    
    def ceiloffive(x, base=5):
        return base * math.ceil(x/base)
    
    #generate specific strings
    if (plat>0):
        northing_num=flooroffive(platabs, base=5)
    else: 
       northing_num=ceiloffive(platabs, base=5) 
    northing_str=str(northing_num).zfill(2)
    
    if (plon>0):
        easting_num=flooroffive(plonabs, base=5)
    else:
         easting_num=ceiloffive(plonabs, base=5)
    easting_str=str(easting_num).zfill(3)
    
    filename_only=file_north+northing_str+file_hemi+easting_str+file_suffix
    
    filename_w_path=os.path.join(folder_initial,foldername,filename_only)
    
    return filename_w_path
