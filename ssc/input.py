"""
Read in the json of S3 links, picks the correct one, reads it in, and finds what reaches and nodes we will predict on for this run.

"""

# Standard imports
import os
import json
import requests
import glob
import time
import random
import subprocess as sp
import logging

# Third-party imports
import botocore
import pandas as pd
import geopandas as gpd
import netCDF4 as ncf
from geopandas.tools import sjoin
from shapely.geometry import Polygon, LineString, MultiPoint, Point
import rioxarray
import boto3
from rasterio.session import AWSSession
import rasterio as rio
import earthaccess
import mgrs #conda install -c conda-forge mgrs
import datetime
import numpy as np


# Local imports

# Functions

def given_tile_geometry_find_nodes(tile_geometry, sword_files, latlon_file):
    """
    function descripotion
    """
    # print(tile_geometry, 'tile geo')
    node_ids_reach_ids_lat_lons = []

    # if latlon_file is not None:
    with open(latlon_file) as json_file:
        all_node_points_f = json.load(json_file)
    # all_node_lat_lons = [(float(i.split(',')[0]),float(i.split(',')[1])) for i in all_node_points_f]
    #these are still in lat lon below
    # all_node_points = [Point(i[1], i[0]) for i in all_node_lat_lons]

    # below didnt work for tiny dataset
    # all_node_points = []
    # for reach in all_node_points_f['coordinates']:
    #     for node in reach:
    #         all_node_points.append(Point(node[1],node[0]))

    #all_node_points = list(set(all_node_points))
    # print('here are all node points', all_node_points)

    all_node_lat_lons = [(float(i.split(',')[1]),float(i.split(',')[0])) for i in all_node_points_f]
    all_node_points = [Point(i[0], i[1]) for i in all_node_lat_lons]
    
    
    sword_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry':all_node_points, 
                                    'nodeid':['foo' for i in all_node_points], 
                                    'reachid':['foo' for i in all_node_points]}),
                                        geometry='geometry',
                                        crs='epsg:4326')
    # print('here is sword gdf', sword_gdf)
    sentinal_tiles = sjoin(sword_gdf, tile_geometry, how='inner', op='within')
    # print(sentinal_tiles)
    # print('here is the sjoin df..', sentinal_tiles)
    # exit()
    # time.wait(500)

    node_ids_reach_ids_lat_lons =  list(zip(sentinal_tiles['nodeid'], sentinal_tiles['reachid'], [(i.x, i.y) for i in sentinal_tiles['geometry']]))
    # print('latlon file provided...')
    # print('here are some example node points...', all_node_lat_lons[:5])
        
    # else:
    #     # for sword_filepath in sword_files:
    #     #     try:
    #     #         # print('trying to pull sword..')

    #     #         sword = ncf.Dataset(sword_filepath)

    #     #         all_node_lat_lons = list(zip(sword['nodes']['x'][:], sword['nodes']['y'][:]))
    #     #         all_node_points = [Point(float(i[0]),float(i[1])) for i in all_node_lat_lons]
    #     #         # print(all_node_points)

    #     #         sword_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry':all_node_points, 
    #     #                                                 'nodeid':sword['nodes']['node_id'][:], 
    #     #                                                 'reachid':sword['nodes']['reach_id'][:]}),
    #     #                                                     geometry='geometry',
    #     #                                                     crs='epsg:4326')
                
    #     #         # print(sword_gdf)
    #     #         sentinal_tiles = sjoin(sword_gdf, tile_geometry, how='inner', op='within')
    #     #         # print(sentinal_tiles)

    #     #         node_ids_reach_ids_lat_lons =  list(zip(sentinal_tiles['nodeid'], sentinal_tiles['reachid'], all_node_lat_lons))
    #     #         sword.close()
    #     #     except Exception as e:
    #     #         print(e)
    #     #     if len(node_ids_reach_ids_lat_lons) > 1:
    #     #         print(f'found { len(node_ids_reach_ids_lat_lons)} nodes...')
    #     #         break
        
        

    return node_ids_reach_ids_lat_lons

def find_tile_geometry(tile_code:str, gdf):
    """
    function descripotion
    """
    
    tile_geometry = gdf[gdf['Name'] == tile_code]
    return tile_geometry

def load_bands_to_memory(tile:str, run_location:str):


    band_definitions = {
        'L':["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11"],
        'S':["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    }




    l_or_s = detect_landsat_or_sentinal(tile)

    #load to memory
    rio_env = login_to_s3()
    all_tiles_in_memory = []

    model_bands = []
    

    for band_suffix in band_definitions[l_or_s]:
        full_s3_band_path = tile+ '.0.' + band_suffix + '.tif' # this is the right one
        #full_s3_band_path = tile + '.'+band_suffix + '.tif'
        # full_s3_band_path = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T38NKF.2018023T072159.v2.0/HLS.S30.T38NKF.2018023T072159.v2.0.B09.tif'
        # hls_da = rioxarray.open_rasterio(full_s3_band_path, chuncks=True)
        http_prefix = 'https://data.lpdaac.earthdatacloud.nasa.gov/'
        # http_prefix = 'https://archive.swot.podaac.earthdata.nasa.gov/'
        s3_prefix = 's3://'
        if run_location == 'aws':
            full_s3_band_path = full_s3_band_path.replace(http_prefix, s3_prefix)
        # full_s3_band_path = full_s3_band_path.replace('protected', 'public')
        # print('here is path', full_s3_band_path)
        # hls_da = rio.open(full_s3_band_path, chuncks=True)
        # sp.run(['wget', full_s3_band_path, "-P", "/data/input/ssc/tmp"])

        hls_da = rioxarray.open_rasterio(full_s3_band_path, chunks=True)[0]
        # hls_da = rioxarray.open_rasterio(os.path.join('/data/input/ssc/tmp',os.path.basename(full_s3_band_path)), chuncks=True)
        all_tiles_in_memory.append(hls_da)
        # sp.run(['rm',  os.path.join('/data/input/ssc/tmp',os.path.basename(full_s3_band_path))])
    
        if l_or_s == 'S':
            tile_code = os.path.basename(full_s3_band_path).split('.')[2][1:]
            # hls_da_for_code = rioxarray.open_rasterio(full_s3_band_path, chuncks=True)
            cloud_cover = hls_da.cloud_coverage
        if l_or_s == 'L':
            # hls_da_for_code = rioxarray.open_rasterio(full_s3_band_path, chuncks=True)
            tile_code = hls_da.SENTINEL2_TILEID
            cloud_cover = hls_da.cloud_coverage
            
        hls_da.close()
        
        if cloud_cover > 50:
            logging.info(f'Cloud cover too high for tile at {cloud_cover}%...')
            logging.info('exiting...')
            # exit()
            raise ValueError(f'Cloud cover too high for tile at {cloud_cover}%...')


    
    # print('tile code test')
    date = datetime.datetime.strptime(os.path.basename(tile).split('.')[3].split('T')[0], '%Y%j').date().strftime('%Y-%m-%d')
  
    rio_env.__exit__()
    return all_tiles_in_memory, l_or_s, tile_code, cloud_cover, date

    # hls_da = rioxarray.open_rasterio(s3_link, chuncks=True)

def detect_landsat_or_sentinal(tile:str):
    """
    Takes in a HLS S3 filepath and determines if it is a landast or sentinal tile
    """
    return os.path.basename(tile).split('.')[1][0]

def get_creds_to_env():
    """Return AWS S3 credentials to access S3 shapefiles."""

    creds = {}
    try:
        ssm = boto3.client("ssm", region_name="us-west-2")
        prefix = os.environ["PREFIX"]
        creds['user'] = ssm.get_parameter(Name=f'{prefix}-lpdaac-user', WithDecryption=True)['Parameter']['Value']
        creds['pass'] = ssm.get_parameter(Name=f'{prefix}-lpdaac-password', WithDecryption=True)['Parameter']['Value']
    except KeyError as e:

        logging.info('Using local .netrc file; this needs to be created prior to running')
        logging.info(f'Here is why we are not using the netrc {e}')
    except botocore.exceptions.ClientError as e:
        raise e

    if creds:
        logging.info('creating netrc')
        f = open("/root/.netrc", "w")
        f.write(f"machine urs.earthdata.nasa.gov login {creds['user']} password {creds['pass']}")
        f.close()

def login_to_s3():
    get_creds_to_env()
    auth = earthaccess.login(strategy="netrc")
    # auth = earthaccess.login(strategy="environment")
    s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'
    temp_creds_url = s3_cred_endpoint
    headers = {'User-Agent': 'Mozilla/5.0'}
    temp_creds_req = requests.get(temp_creds_url, headers=headers)
    # print(temp_creds_req.content)
    try:
        temp_creds_req = temp_creds_req.json()
    except Exception as e:
        logging.info(e)
        logging.info('temp creds request failed')
        # print('here is raw response for bugfixing...', temp_creds_req.content)
        # exit()
        raise ValueError('temp creds request failed')
    

    session = boto3.Session(aws_access_key_id=temp_creds_req['accessKeyId'], 
                        aws_secret_access_key=temp_creds_req['secretAccessKey'],
                        aws_session_token=temp_creds_req['sessionToken'],
                        region_name='us-west-2')
    # session = boto3.Session(region_name='us-west-2')

    # client = boto3.client('sts')
    # response = client.get_session_token()
    # session = boto3.Session(aws_access_key_id=response["Credentials"]["AccessKeyId"], 
    #                     aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
    #                     aws_session_token=response["Credentials"]["SessionToken"],
    #                     region_name='us-west-2')

    rio_env = rio.Env(AWSSession(session),
                  GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                  GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                  GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))
    rio_env.__enter__()
    return rio_env



def input(index_to_run:int, json_data , sentinel_shapefile_filepath:str, latlon_file, run_location:str, reaches_of_interest_path:str, sword_dir:str, sword_data):
    """
    function descripotion
    """




    # tile_code = tile_filename.split('.')[2]

    # read in gdf of sentinal

    if latlon_file is not None:
        
        tile = list(json_data.keys())[index_to_run]
        
        # all_tiles_in_memory, l_or_s, tile_code, cloud_cover, date
        all_bands_in_memory, l_or_s, tile_code, cloud_cover, date = load_bands_to_memory(tile = tile, run_location = run_location)

        # parse filename for tile code and filename for return
        tile_filename = os.path.basename(tile)

        gdf = gpd.read_file(sentinel_shapefile_filepath)
        
        all_swords = glob.glob(os.path.join(sword_dir, '*.nc'))

        # find reaches and nodes to predict on
        tile_geometry = find_tile_geometry(tile_code=tile_code, gdf=gdf)

        node_ids_reach_ids_lat_lons = given_tile_geometry_find_nodes(tile_geometry, all_swords, latlon_file)
    
    else:
        tile = list(json_data.keys())[index_to_run]
        
        # all_tiles_in_memory, l_or_s, tile_code, cloud_cover, date
        all_bands_in_memory, l_or_s, tile_code, cloud_cover, date = load_bands_to_memory(tile = tile, run_location = run_location)

        # parse filename for tile code and filename for return
        tile_filename = os.path.basename(tile)
        
        tile_reaches = json_data[tile]
        
        
        if reaches_of_interest_path is not None:
            overlapping_reaches = []
            with open(reaches_of_interest_path) as f:
                reaches_of_interest = json.load(f)
            reaches_of_interest = [str(i) for i in reaches_of_interest]
            for a_tile_reach in tile_reaches:
                if str(a_tile_reach) in reaches_of_interest:
                    overlapping_reaches.append(a_tile_reach)
        else:
            overlapping_reaches = tile_reaches
            


        # overlapping_val_reaches = []

        # for i in overlapping_reaches:
        #     if os.path.exists(os.path.join(validation_dir, f'{i}_validation.nc')):
        #         overlapping_val_reaches.append(i)

        # overlapping_reaches = overlapping_val_reaches

        if len(overlapping_reaches) == 0:
            print('no reaches found in tile, exiting...')
            # exit()
            raise ValueError('no reaches found in tile, exiting...')
        node_ids_reach_ids_lat_lons = given_reach_find_nodes(overlapping_reaches = overlapping_reaches, sword_data = sword_data)
    # print(node_ids_reach_ids_lat_lons, 'here are points')
    # node_ids_reach_ids_lat_lons = node_ids_reach_ids_lat_lons[:4]

    # return bands in memory for preprocessing, and processing targets
    return all_bands_in_memory, node_ids_reach_ids_lat_lons, tile_filename, l_or_s, tile_code, cloud_cover, date

def load_correct_sword(a_reach:str, sword_dir:str):
    lookup = {
    "1": "af",
    "4": "as", "3": "as",
    "2": "eu",
    "7": "na", "8": "na", "9": "na",
    "5": "oc",
    "6": "sa"
    }
    
    sword_path = os.path.join(sword_dir, f'{lookup[a_reach[0]]}_sword_v16_patch.nc')

    if os.path.exists(sword_path):
        pass
    else:
        sword_path = sword_path.replace('_patch', '')

    
    return ncf.Dataset(sword_path)
    

def given_reach_find_nodes(overlapping_reaches:list, sword_data):

    lat_list, lon_list , node_ids, reach_ids= [], [], [], []

    # sword_fp = os.path.join(data_dir, f'{cont.lower()}_sword_v15.nc')
    # print(f'Searching across {len(files)} continents for nodes...')
    
    

    for reach_id in overlapping_reaches:
        node_ids_indexes = np.where(sword_data.groups['nodes'].variables['reach_id'][:].data.astype('U') == str(reach_id))

        if len(node_ids_indexes[0])!=0:
            for y in node_ids_indexes[0]:

                lat = float(sword_data.groups['nodes'].variables['x'][y].data.astype('U'))
                lon = float(sword_data.groups['nodes'].variables['y'][y].data.astype('U'))
                node_id = sword_data.groups['nodes'].variables['node_id'][y]
                # all_nodes.append([lat,lon])
                lat_list.append(lat)
                lon_list.append(lon)
                node_ids.append(node_id)
                reach_ids.append(reach_id)
            


    
    node_ids_reach_ids_lat_lons = [[a, b, (c, d)] for a, b, c, d in zip(node_ids, reach_ids, lat_list, lon_list)]


    # print(f'Found {len(all_nodes)} nodes...')
    return node_ids_reach_ids_lat_lons
    

def get_region(num):
    return lookup.get(num)
    

def mgrs_flag_generation(mgrs_code, node_data):
    pointlat = node_data[2][1]
    pointlon = node_data[2][0]
    m_mgrs = mgrs.MGRS()
    mgrs_coords = m_mgrs.toMGRS(pointlat, pointlon)
    mgrs_coords = mgrs_coords[:5]
    if mgrs_coords == mgrs_code:
        mgrs_flag = 0
    else:
        mgrs_flag = 1

    return mgrs_flag

