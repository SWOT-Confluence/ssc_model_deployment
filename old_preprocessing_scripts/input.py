"""
Read in the json of S3 links, picks the correct one, reads it in, and finds what reaches and nodes we will predict on for this run.

"""

# Standard imports
import os
import json
import requests

# Third-party imports
import pandas as pd
import geopandas as gpd
import netCDF4 as ncf
from geopandas.tools import sjoin
from shapely.geometry import Polygon, LineString, MultiPoint, Point
import rioxarray
import boto3
from rasterio.session import AWSSession
import rasterio as rio

# Local imports

# Functions

def given_tile_geometry_find_nodes(tile_geometry, sword):
    """
    function descripotion
    """

    all_node_lat_lons = list(zip(sword['nodes']['x'][:], sword['nodes']['y'][:]))
    all_node_points = [Point(float(i[0]),float(i[1])) for i in all_node_lat_lons]
    sword_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry':all_node_points, 
                                               'nodeid':sword['nodes']['node_id'][:], 
                                               'reachid':sword['nodes']['reach_id'][:]}),
                                                geometry='geometry',
                                                crs='epsg:4326')
    sentinal_tiles = sjoin(sword_gdf, tile_geometry, how='inner', op='within')

    nids_rids = list(zip(sentinal_tiles['nodeid'], sentinal_tiles['reachid']))

    return nids_rids

def find_tile_geometry(tile_code:str, gdf):
    """
    function descripotion
    """
    
    tile_geometry = gdf[gdf['Name'] == tile_code]
    print('Found tile geometry')
    return tile_geometry

def load_tiles_to_memory(tile:str):

    band_definitions = {
        'S':[".B02.tif",
            ".B01.tif",
            ".B03.tif",
            ".B04.tif",
            ".B05.tif",
            ".B06.tif",
            ".B07.tif",
            ".B08.tif",
            ".B8A.tif",
            ".B09.tif",
            ".B10.tif",
            ".B11.tif",
            ".B12.tif",],

        'L':['.B01.tif',
            '.B02.tif',
            '.B03.tif',
            '.B04.tif',
            '.B05.tif',
            '.B06.tif',
            '.B07.tif',
            '.B09.tif',
            '.B10.tif',
            '.B11.tif',]
    }

    l_or_s = detect_landsat_or_sentinal(tile)
    # print(os.path.basename(tile))

    #load to memory
    rio_env = login_to_s3()
    

    all_tiles_in_memory = []

    for band_suffix in band_definitions[l_or_s]:
        full_s3_band_path = os.path.join(tile, os.path.basename(tile)+band_suffix)
        print(full_s3_band_path)
        # full_s3_band_path = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T38NKF.2018023T072159.v2.0/HLS.S30.T38NKF.2018023T072159.v2.0.B09.tif'
        hls_da = rioxarray.open_rasterio(full_s3_band_path, chuncks=True)
        all_tiles_in_memory.append(hls_da)
    #     print(hls_da)
    rio_env.__exit__()

    return all_tiles_in_memory, l_or_s

        
    



    # hls_da = rioxarray.open_rasterio(s3_link, chuncks=True)

def detect_landsat_or_sentinal(tile:str):
    """
    Takes in a HLS S3 filepath and determines if it is a landast or sentinal tile
    """
    return os.path.basename(tile).split('.')[1][0]



def login_to_s3():
    s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'
    temp_creds_url = s3_cred_endpoint
    print(temp_creds_url)
    headers = {'User-Agent': 'Mozilla/5.0'}
    temp_creds_req = requests.get(temp_creds_url, headers=headers)
    print(temp_creds_req.text)
    temp_creds_req = temp_creds_req.json()
    session = boto3.Session(aws_access_key_id=temp_creds_req['accessKeyId'], 
                        aws_secret_access_key=temp_creds_req['secretAccessKey'],
                        aws_session_token=temp_creds_req['sessionToken'],
                        region_name='us-west-2')
    rio_env = rio.Env(AWSSession(session),
                  GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                  GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                  GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))
    rio_env.__enter__()
    return rio_env



def input(indir:str, index_to_run:int, hls_s3_json_filename:str , sentinal_shapefile_filepath:str, sword_filepath:str):
    """
    function descripotion
    """

    # parse input filename
    hls_s3_json_filename = os.path.join(indir, hls_s3_json_filename)

    # read in hls_s3_json_path
    with open(hls_s3_json_filename) as f:
        json_data = json.load(f)
        # print(d)

    # args.index to select the correct tile
    tile = json_data[index_to_run]

    # PORT FROM PREPROCESSING
    # download and load in for return
    all_tiles_in_memory, l_or_s = load_tiles_to_memory(tile)

    # parse filename for tile code and filename for return
    tile_filename = os.path.basename(tile)
    tile_code = all_tiles_in_memory[0].SENTINEL2_TILEID

    # tile_code = tile_filename.split('.')[2]
    print('TILECODE', tile_code)

    # read in gdf of sentinal
    gdf = gpd.read_file(sentinal_shapefile_filepath)

    # find what sword to use

    # read in sword
    sword = ncf.Dataset(sword_filepath)

    # find reaches and nodes to predict on
    tile_geometry = find_tile_geometry(tile_code=tile_code, gdf=gdf)
    node_ids_reach_ids = given_tile_geometry_find_nodes(tile_geometry, sword)

    # return bands in memory for preprocessing, and processing targets
    return all_tiles_in_memory, node_ids_reach_ids, tile_filename, l_or_s


