# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:35:13 2023

@author: Luisa Lucchese

Building HLS filenames based on date

inputs:
    - date: the actual date of the HLS
    - hour_of_acquisition: hour of acquisition of the HLS
    - pointlon: longitude of the in situ point
    - pointlat: latitude of the in situ point
    - path_to_file: folder where the HLS file can be found
    - band: the band to search for (1 to 12)

outputs:
    - HLS_filename: the generated HLS filename

"""

def build_HLS_filenames(date,hour_of_acquisition,pointlon,pointlat,path_to_file,band, source):
    
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
    
    # pre1='HLS.S30.T'
    # post1='.subset.tif'
    pre1 = f'HLS.{source}30.T'
    post1='_mult.tif'
    version='.v2.0.'
    band_compose='B'+str(band).zfill(2)
    #date_compose=year+str(juliandate_year).zfill(3) #uncomment if date comes in YY-MM-DD format
    
    HLS_filename=path_to_file+'/'+pre1+tile+'.'+date_compose+'T'+hour_of_acquisition+version+band_compose+post1

    return HLS_filename

