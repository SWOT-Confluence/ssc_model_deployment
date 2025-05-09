# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:40:57 2023

@author: Luisa Lucchese

Build HLS filenames for the extra bands

inputs:
    - date: the actual date of the HLS
    - hour_of_acquisition: hour of acquisition of the HLS
    - pointlon: longitude of the in situ point
    - pointlat: latitude of the in situ point
    - path_to_file: folder where the HLS file can be found
    - bandname: a string with the name of the band to be read
outputs:
    - HLS_filename: the generated HLS filename
"""

def build_HLS_filenames_extrabands(date,hour_of_acquisition,pointlon,pointlat,path_to_file,source,bandname):
    
    # date='2021-09-07'
    # band=1
    # hour_of_acquisition='154809' #HHMMSS
    # path_to_file='D:/Luisa/start_insitu_data_wrongdata/downloaded'
    # pointlon=-73.9978
    # pointlat=40.23028
    #T60HTE– MGRS Tile ID (T+5-digits)
    #tile='18TWK'
    #UTM Zone 19
    #Latitude Band T
    #MGRS column D
    #MGRS row J
    
    #example
    #HLS.S30.T18TWK.2021250T154809.v2.0.B01.subset.tif
    # naming convention at https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/#hls-naming-conventions
    
    #import datetime
    import mgrs #conda install -c conda-forge mgrs
    
    m_mgrs = mgrs.MGRS()
    mgrs_coords = m_mgrs.toMGRS(pointlat, pointlon)
    tile = mgrs_coords[:5]
    
    date_compose=date #comment if date comes in YY-MM-DD format
    #[year,month,day]=date.split('-') #uncomment if date comes in YY-MM-DD format
    #juliandate_total=datetime.date(int(year),int(month),int(day)).toordinal() #uncomment if date comes in YY-MM-DD format
    #juliandate_year=juliandate_total-datetime.date(int(year),int(1),int(1)).toordinal()+1 #uncomment if date comes in YY-MM-DD format
    
    pre1=f'HLS.{source}30.T'#'HLS.S30.T'
    post1='_mult.tif'#'.subset.tif'
    version='.v2.0.'
    band_compose=bandname
    #date_compose=year+str(juliandate_year).zfill(3) #uncomment if date comes in YY-MM-DD format
    
    HLS_filename=path_to_file+'/'+pre1+tile+'.'+date_compose+'T'+hour_of_acquisition+version+band_compose+post1

    return HLS_filename


"""
Created on Thu Feb 23 10:40:57 2023

@author: Luisa Lucchese

Build HLS filenames for the extra bands, using just the HLS name to build the
new name

inputs:
    - filename: original filename
    - bandname: a string with the name of the band to be read
    - numchar: number of characters to be removed from the name of the file
outputs:
    - HLS_filename: the generated HLS filename
"""


def build_HLS_filenames_extrabands_namepres(filename,bandname,numchar):
    
    # preserving the name of the file
    
    # date='2021-09-07'
    # band=1
    # hour_of_acquisition='154809' #HHMMSS
    # path_to_file='D:/Luisa/start_insitu_data_wrongdata/downloaded'
    # pointlon=-73.9978
    # pointlat=40.23028
    #T60HTE– MGRS Tile ID (T+5-digits)
    #tile='18TWK'
    #UTM Zone 19
    #Latitude Band T
    #MGRS column D
    #MGRS row J
    
    #example
    #HLS.S30.T18TWK.2021250T154809.v2.0.B01.subset.tif
    # naming convention at https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/#hls-naming-conventions
    
    post1='_mult.tif'#'.subset.tif'

    #date_compose=year+str(juliandate_year).zfill(3) #uncomment if date comes in YY-MM-DD format
    filename = '.'.join(filename.split('.')[:-2])+'.'
    HLS_filename=filename+bandname+post1

    return HLS_filename
