"""
Description

"""

# Standard imports
import glob
import sys
import os
import requests
import json
# from datetime import datetime, timedelta
# import multiprocessing
import glob
import time as tm

# Third-party imports
# import boto3
import rasterio as rio
from rasterio.session import AWSSession
import pandas as pd
from loguru import logger
# import tifffile
import numpy as np
import cv2
import rasterio
# import osr
from osgeo import gdal

# Local imports
from ssc.preprocessing_functions.ssc_preprocessing_functions.deepwatermap import model as deepwatermap
from ssc.preprocessing_functions.ssc_preprocessing_functions.extract_attributes_HLS_v1 import extract_value
from ssc.preprocessing_functions.ssc_preprocessing_functions.build_HLS_filenames import build_HLS_filenames
from ssc.preprocessing_functions.ssc_preprocessing_functions.build_HLS_filenames_extrabands import build_HLS_filenames_extrabands,build_HLS_filenames_extrabands_namepres
from ssc.preprocessing_functions.ssc_preprocessing_functions.extract_info_HLS_filename import extract_info_HLS_filename
from ssc.preprocessing_functions.ssc_preprocessing_functions.function_sunmask import function_sunmask
from ssc.preprocessing_functions.ssc_preprocessing_functions.reproject_sunmask import reproject_sunmask
from ssc.preprocessing_functions.ssc_preprocessing_functions.buildfilename_MERIT_DEM import buildfilename_MERIT_DEM
from ssc.preprocessing_functions.ssc_preprocessing_functions.reproject_to_UTM import reproject_to_UTM
from ssc.preprocessing_functions.ssc_preprocessing_functions.reproject_HLS import reproject_HLS
from ssc.preprocessing_functions.ssc_preprocessing_functions.read_Fmask import read_Fmask
from ssc.preprocessing_functions.ssc_preprocessing_functions.crop_HLS_save import crop_HLS_save,crop_HLS_save_flagMGRS1
from ssc.preprocessing_functions.ssc_preprocessing_functions.recognize_landsat_sentinel import recognize_landsat_sentinel

# Functions
def get_temp_creds():
    temp_creds_url = s3_cred_endpoint
    return requests.get(temp_creds_url).json()

def load_data(csv_fp, json_fp):
    # TODO: possible issue is if csv_data or json_data is too large to be loaded to memory
    logger.debug(f"Loading data from {csv_fp} and {json_fp}")
    csv_data = pd.read_csv(csv_fp)
    with open(json_fp, 'r') as f:
        json_data = json.load(f)
    logger.info(f"Loaded json: {len(json_data)} and csv:{csv_data.shape}")
    
    return csv_data, json_data

def save_hls_data(hls_links, fp_dir_out):
    for url in hls_links:
        # TODO: use time from aqsat to filter the links (i.e., get data closest to aqsat `date_utc`)
        # TODO: need to add time in json file if needed (since there can be multiple tiles in a day)
        logger.debug(f"Getting data from: {url}")

        # send a HTTP request and save
        fp_out = os.path.join(fp_dir_out, url.split("/")[-1])
        if not os.path.exists(fp_out):
            r = requests.get(url) # create HTTP response object
            if r.content == b'HTTP Basic: Access denied.\n':
                raise PermissionError
            with open(fp_out,'wb') as f:
                f.write(r.content)

def get_hls_links(json_data, site_id, date, fail_log_fp=FAIL_LOG_FP):
    try:
        hls_links = json_data[site_id]["dates"][date]['links']
        extra_bands = ['Fmask', 'SAA', 'SZA', 'VAA', 'VZA']
        for name in extra_bands:
            print(hls_links)
            name_link_list = hls_links[0].split('.')
            name_link_list[-2] = name
            name_link = '.'.join(name_link_list)
            # print(name_link)
            hls_links.append(name_link)
        if not isinstance(hls_links, list):
            logger.error(f"KeyError: [{site_id}, {date}] not found in `json_data`")
            return []    
    except KeyError:
        logger.error(f"KeyError: [{site_id}, {date}] not found in `json_data`")
        with open(fail_log_fp, "a") as fp:
            fp.write(f"[{site_id}, {date}]: not found in `json_data`,\n")
        return [] # return no data
    
    # raise
    return hls_links


def process_row(row, json_data, out_dir):
    # NOTE: this is the function that can be called in parallel
    site_id = row[SITE_ID_COL]

    # Get all dates, +/-1 day from listed day (including original date)
    date = row['date']
    fp_dir_out = os.path.join(out_dir, site_id, date)

    if not False:
        outfiles = glob.glob(os.path.join(fp_dir_out, '*.csv'))
        if len(outfiles) > 0:
            raise ValueError('Site allready processed')
    
    if not os.path.exists(os.path.join(out_dir, site_id)):          # make folder with site id
        os.makedirs(os.path.join(out_dir, site_id))
    if not os.path.exists(os.path.join(out_dir, site_id, date)):    # make folder with date as name
        os.makedirs(os.path.join(out_dir, site_id, date))
    hls_links = get_hls_links(json_data, site_id, date)


    
    save_hls_data(hls_links, fp_dir_out)
    return site_id, date, fp_dir_out


def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def get_water_mask(model_path, image, out_dir, mask_thresh=0.5):
    """
    Params:
        - model_path: path to the trained model checkpoint
        - image: multispectral image composed of bands 2 to 7 from HLS (size: 6,h,w)
        - out_dir: where to save image mask if masks are to be saved
    Returns: water mask array (1 for water pixels, np.nan for non-water pixels)
    """
    # load the model
    logger.debug(f"mask_thresh: {mask_thresh}")
    model = deepwatermap.model()
    model.load_weights(model_path).expect_partial()    # see https://github.com/tensorflow/tensorflow/issues/43554

    image = np.transpose(image, (1,2,0))
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')

    # solve no-pad index issue after inference
    if pad_r[1] == 0:
        pad_r = (pad_r[0], 1)
    if pad_c[1] == 0:
        pad_c = (pad_c[0], 1)

    image = image.astype(np.float32)

    # remove nans (and infinity) - replace with 0s
    image = np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    pred_start_time = tm.time()
    dwm = model.predict(image)
    pred_time = tm.time() - pred_start_time
    dwm = np.squeeze(dwm)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    dwm = 1./(1+np.exp(-(16*(dwm-0.5))))
    dwm = np.clip(dwm, 0, 1)
    
    water_mask = np.where(dwm > mask_thresh, 1, np.nan)
    if TO_SAVE_MASKS:
        thresh_dwm = np.where(dwm > mask_thresh, 255, 0)
        cv2.imwrite(os.path.join(out_dir, "water_mask_out.png"), thresh_dwm)
    return water_mask, pred_time

def remove_all_non_cropped_imgs(row_outpath):
    all_non_cropped_imgs = glob.glob(os.path.join(row_outpath, '*.tif'))
    for img in all_non_cropped_imgs:
        os.remove(img)

"""
Links to replace:

base_dir = '/mnt/data'
s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'
FAIL_LOG_FP = os.path.join(base_dir, 'logs',"failure_indices.txt") # where failed runs will be logged
DATA_DIR = os.path.join(base_dir, 'input')    # where data is stored (csv, json reference files)
DEM_DIR = os.path.join(DATA_DIR, 'merit_dem')
OUT_DIR = os.path.join(base_dir, 'output')  # folder where data will be downloaded to
MODEL_CKPT_DIR = os.path.join(base_dir, 'checkpoint')
# ssc_json_path = os.path.join(DATA_DIR, 'checkpoint_197000.json')
ssc_json_path = os.path.join(DATA_DIR, 'final.json')

# aqsat_path = os.path.join(DATA_DIR, 'filtered_data.csv')    # filtered data from 00_filter_csv.py
# aqsat_path = os.path.join(DATA_DIR, 'preprocessing_targets.csv')    # filtered data from 00_filter_csv.py
aqsat_path = os.path.join(DATA_DIR, '51873_sampled_tss_ssc_preprocessing_targets.csv')

MODEL_CKPT_PATH = os.path.join(base_dir, 'checkpoint', 'cp.135.ckpt')

output_raster_intermed = os.path.join(INTERMED_OUT_DIR, 'intermed1.tif')
DEM_intermed = os.path.join(INTERMED_OUT_DIR, 'merit_reproj_inter.tif')
dst_filepath = os.path.join(INTERMED_OUT_DIR ,'shadow_python_15.tif')
dst_filename_wos = os.path.join(INTERMED_OUT_DIR, 'intermed_wos.tif')
dst_filename_sun = os.path.join(INTERMED_OUT_DIR, 'intermed_sun.tif')
maskedout = os.path.join(INTERMED_OUT_DIR,'maskedout.tif')
DEM = os.path.join(INTERMED_OUT_DIR,'merit_reproj.tif')

"""

def cv_preprocessing(indir, save_masks_bool, max_thresh_for_multitasking_vision_model_preprocessing):
        # will be changed to mount command
    # base_dir = '/mnt/data'
    # base_dir = '/media/travis/work/data/ssc'

    # Configureations bools
    TO_SAVE_MASKS = save_masks_bool   # Set to True if masks should be saved for visual checks

    # PODAAC S3 Bucket Credentials Endpoint for HLS Download
    s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'

    # Directory structure
    FAIL_LOG_FP = os.path.join(indir, 'logs',"failure_indices.txt") # where failed runs will be logged
    DATA_DIR = os.path.join(indir, 'input')    # where data is stored (csv, json reference files)
    DEM_DIR = os.path.join(DATA_DIR, 'merit_dem')
    OUT_DIR = os.path.join(indir, 'output')  # folder where data will be downloaded to
    MODEL_CKPT_DIR = os.path.join(indir, 'checkpoint')

    # INTERMED_OUT_DIR = os.path.join(OUT_DIR, 'intermediate')
    # output_raster_intermed = os.path.join(INTERMED_OUT_DIR, 'intermed1.tif')
    # DEM_intermed = os.path.join(DATA_DIR, 'MERIT_DEM_UTM', 'merit_reproj_inter.tif')
    # dst_filepath = os.path.join(base_dir ,'shadow_test','shadow_python_15.tif')
    # dst_filename = os.path.join(INTERMED_OUT_DIR, 'intermed25.tif')
    # maskedout = os.path.join(OUT_DIR,'maskedout.tif')

    #intermediate local products




    # Filenames

    # DEM = os.path.join(DATA_DIR, 'MERIT_DEM_UTM','merit_reproj.tif')



    # ssc_json_path = os.path.join(DATA_DIR, 'checkpoint_197000.json')
    ssc_json_path = os.path.join(DATA_DIR, 'final.json')

    # aqsat_path = os.path.join(DATA_DIR, 'filtered_data.csv')    # filtered data from 00_filter_csv.py
    # aqsat_path = os.path.join(DATA_DIR, 'preprocessing_targets.csv')    # filtered data from 00_filter_csv.py
    aqsat_path = os.path.join(DATA_DIR, '51873_sampled_tss_ssc_preprocessing_targets.csv')

    MODEL_CKPT_PATH = os.path.join(base_dir, 'checkpoint', 'cp.135.ckpt')

    # Input data csv headers
    DATE_STR_FMT = '%Y-%m-%d'
    LAT_COL = "lat"     # column name for latitude
    LON_COL = "lon"    # column name for longitude
    SITE_ID_COL = "SiteID"  # column name for Site ID

# def main():

    # Command line arguments
    arg_parser = create_args()
    args = arg_parser.parse_args()
    
    index = args.index
    overall_start_time = tm.time()

    # read in data
    df = pd.read_csv(aqsat_path)

    with open(ssc_json_path, "r") as read_file:
        json_data = json.load(read_file)

    # download tile
    retry = 0

    # variables used for time benchmarks
    dl_time = 0
    img_loading_time = 0
    water_mask_time = 0
    total_pred_time = 0
    water_mask_mult_time = 0
    dem_shadow_time = 0
    fmask_shadow_time = 0
    feat_extract_time = 0
    save_to_csv_time = 0
    save_cropped_imgs_time = 0
    remove_large_imgs_time = 0
    while retry <= 3:
        try:
            start_time = tm.time()
            print(f'processing row for the {retry+1} time')
            site_id, date, row_outpath = process_row(df.iloc[index], json_data, OUT_DIR)
            print('Images downloaded successfully...')
            dl_time += tm.time() - start_time

            INTERMED_OUT_DIR = row_outpath
            output_raster_intermed = os.path.join(INTERMED_OUT_DIR, 'intermed1.tif')
            DEM_intermed = os.path.join(INTERMED_OUT_DIR, 'merit_reproj_inter.tif')
            dst_filepath = os.path.join(INTERMED_OUT_DIR ,'shadow_python_15.tif')
            dst_filename_wos = os.path.join(INTERMED_OUT_DIR, 'intermed_wos.tif')
            dst_filename_sun = os.path.join(INTERMED_OUT_DIR, 'intermed_sun.tif')
            maskedout = os.path.join(INTERMED_OUT_DIR,'maskedout.tif')
            DEM = os.path.join(INTERMED_OUT_DIR,'merit_reproj.tif')
            # print('bands')
            # part to recognize landsat or sentinel before running water mask - Luisa June 2023
            band_paths = glob.glob(os.path.join(row_outpath, '*1.tif'))
            # print('1')
            # print(band_paths)
            filename_early = band_paths[0]
            # print('2')
            ncrop=7 #number of characters to crop from the end of the filename, removing
            # print('ok')
            actual_filename, path_to_file, band, tile, date, time, version, source=extract_info_HLS_filename(filename_early)
            # recognize whether it is from Sentinel or Landsat
            # print('recognizing')
            # source=recognize_landsat_sentinel(actual_filename)
            # end of the part to recognize or sentinel
            # print('recognized')
            #produce water mask
            bands = [ '.'.join(i.split('.')[:-2])+'.' for i in glob.glob(os.path.join(row_outpath, '*'))]
            # print(bands)
            # print('yuee')
            start_time = tm.time()
            for ctr, image_path in enumerate(bands):
                # load and preprocess the input image (might need to change this depending on the file format)
                # only need bands 2 to 7 for HLS dataset
                if image_path != '.':
                    # print('where')
                    try:
                        # print('are')
                        if image_path.endswith("."):
                            # print('we')
                            if source=='S':
                                # print('Proce')
                                B02 = rasterio.open(f'{image_path}B02.tif').read()
                                B01 = rasterio.open(f'{image_path}B01.tif').read()
                                B03 = rasterio.open(f'{image_path}B03.tif').read()
                                B04 = rasterio.open(f'{image_path}B04.tif').read()
                                B05 = rasterio.open(f'{image_path}B05.tif').read()
                                B06 = rasterio.open(f'{image_path}B06.tif').read()
                                B07 = rasterio.open(f'{image_path}B07.tif').read()
                                B08 = rasterio.open(f'{image_path}B08.tif').read()
                                B8A = rasterio.open(f'{image_path}B8A.tif').read()
                                B09 = rasterio.open(f'{image_path}B09.tif').read()
                                B10 = rasterio.open(f'{image_path}B10.tif').read()
                                B11 = rasterio.open(f'{image_path}B11.tif').read()
                                B12 = rasterio.open(f'{image_path}B12.tif').read()
                            elif source=='L':
                                # print('l1')
                                B01 = rasterio.open(f'{image_path}B01.tif').read()
                                B02 = rasterio.open(f'{image_path}B02.tif').read()
                                B03 = rasterio.open(f'{image_path}B03.tif').read()
                                B04 = rasterio.open(f'{image_path}B04.tif').read()
                                B8A = rasterio.open(f'{image_path}B05.tif').read()
                                B11 = rasterio.open(f'{image_path}B06.tif').read()
                                B12 = rasterio.open(f'{image_path}B07.tif').read()
                                B10 = rasterio.open(f'{image_path}B09.tif').read()
                                B13 = rasterio.open(f'{image_path}B10.tif').read()
                                B14 = rasterio.open(f'{image_path}B11.tif').read()



                                
                            #open one of them in gdal just to get the georeference
                            get_geotran = gdal.Open(f'{image_path}B01.tif', gdal.GA_ReadOnly)
                            
                            #get geotransformation
                            geotran= get_geotran.GetGeoTransform()
                            projection_reproduce = get_geotran.GetProjection()
                            #get shape of array
                            getshape_1 = get_geotran.GetRasterBand(1)
                            getshape_2 = getshape_1.ReadAsArray()
                            size1,size2=getshape_2.shape
                        else:
                            # print('not')
                            if source=='S':
                                # print('s1')
                                B01 = rasterio.open(f'{image_path}B1.TIF').read()
                                B02 = rasterio.open(f'{image_path}B2.TIF').read()
                                B03 = rasterio.open(f'{image_path}B3.TIF').read()
                                B04 = rasterio.open(f'{image_path}B4.tif').read()
                                B05 = rasterio.open(f'{image_path}B5.TIF').read()
                                B06 = rasterio.open(f'{image_path}B6.TIF').read()
                                B07 = rasterio.open(f'{image_path}B7.TIF').read()
                                B08 = rasterio.open(f'{image_path}B8.TIF').read()
                                B8A = rasterio.open(f'{image_path}B8A.TIF').read()
                                B09 = rasterio.open(f'{image_path}B9.TIF').read()
                                B10 = rasterio.open(f'{image_path}B10.TIF').read()
                                B11 = rasterio.open(f'{image_path}B11.TIF').read()
                                B12 = rasterio.open(f'{image_path}B12.TIF').read()
                            elif source=='L':
                                # print('l1')
                                B01 = rasterio.open(f'{image_path}B1.TIF').read()
                                B02 = rasterio.open(f'{image_path}B2.TIF').read()
                                B03 = rasterio.open(f'{image_path}B3.TIF').read()
                                B04 = rasterio.open(f'{image_path}B4.TIF').read()
                                B8A = rasterio.open(f'{image_path}B5.TIF').read()
                                B11 = rasterio.open(f'{image_path}B6.TIF').read()
                                B12 = rasterio.open(f'{image_path}B7.TIF').read()
                                B10 = rasterio.open(f'{image_path}B9.TIF').read()
                                B13 = rasterio.open(f'{image_path}B10.TIF').read()
                                B14 = rasterio.open(f'{image_path}B11.TIF').read()
                            #open one of them in gdal just to get the georeference
                            get_geotran = gdal.Open(f'{image_path}B1.TIF', gdal.GA_ReadOnly)
                            #get geotransformation
                            geotran= get_geotran.GetGeoTransform()
                            #get projection
                            projection_reproduce = get_geotran.GetProjection()
                            #get shape of array
                            getshape_1 = get_geotran.GetRasterBand(1)
                            getshape_2 = getshape_1.ReadAsArray()
                            size1,size2=getshape_2.shape
                        break   # stop looping over bands since have already loaded all bands successfully (and to keep the actual image_path value)
                    except:
                        raise ValueError(image_path)
            retry = 999
            img_loading_time += tm.time() - start_time
        except Exception as e:
            print(e)
            # remove non-cropped images
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

            remove_all_non_cropped_imgs(row_outpath=row_outpath)
            retry += 1
    if retry != 999:
        raise ImportError('Failed to download tiles...')
    #image = np.concatenate([B02, B03, B04, B05, B06, B07], axis=0)  # size: (6, h, w) #BASED ON LANDSAT
    image = np.concatenate([B02, B03, B04, B8A, B11, B12], axis=0) #because our band names are now based on sentinel
    start_time = tm.time()
    water_mask, pred_time = get_water_mask(model_path=MODEL_CKPT_PATH, image=image, out_dir=row_outpath, mask_thresh=args.mask_thresh)
    water_mask_time += tm.time() - start_time
    total_pred_time += pred_time

    logger.debug(f"image_path: {image_path}")
    start_time = tm.time()
    # pre-preprocessing: we will need to save the bands
    # multiplied by the binary water mask
    def write_mult_bands(multiplied_array,bandname,size1,size2, geotran,projection_reproduce):
        driver = gdal.GetDriverByName( 'GTiff' )
        dst_ds=driver.Create(f'{image_path}{bandname}_mult.tif',size2,size1,1,gdal.GDT_Float64)
        dst_ds.SetGeoTransform(geotran)
        #srs = osr.SpatialReference()
        dst_ds.SetProjection(projection_reproduce)#srs.ExportToWkt() )
        # print(multiplied_array)
        # print(np.shape(multiplied_array))
        export=dst_ds.GetRasterBand(1).WriteArray(multiplied_array[0])
        export = None
        return None
    
    if source=='S':
        B01_mult=B01#*water_mask
        write_mult_bands(B01_mult,'B01',size1,size2, geotran,projection_reproduce)
        B02_mult=B02#*water_mask
        write_mult_bands(B02_mult,'B02',size1,size2, geotran,projection_reproduce)
        B03_mult=B03#*water_mask
        write_mult_bands(B03_mult,'B03',size1,size2, geotran,projection_reproduce)
        B04_mult=B04#*water_mask
        write_mult_bands(B04_mult,'B04',size1,size2, geotran,projection_reproduce)
        B05_mult=B05#*water_mask
        write_mult_bands(B05_mult,'B05',size1,size2, geotran,projection_reproduce)
        B06_mult=B06#*water_mask
        write_mult_bands(B06_mult,'B06',size1,size2, geotran,projection_reproduce)
        B07_mult=B07#*water_mask
        write_mult_bands(B07_mult,'B07',size1,size2, geotran,projection_reproduce)
        B08_mult=B08#*water_mask
        write_mult_bands(B08_mult,'B08',size1,size2, geotran,projection_reproduce)
        B8A_mult=B8A#*water_mask
        write_mult_bands(B8A_mult,'B8A',size1,size2, geotran,projection_reproduce)
        B09_mult=B09#*water_mask
        write_mult_bands(B09_mult,'B09',size1,size2, geotran,projection_reproduce)
        B10_mult=B10#*water_mask
        write_mult_bands(B10_mult,'B10',size1,size2, geotran,projection_reproduce)
        B11_mult=B11#*water_mask
        write_mult_bands(B11_mult,'B11',size1,size2, geotran,projection_reproduce)
        B12_mult=B12#*water_mask
        write_mult_bands(B12_mult,'B12',size1,size2, geotran,projection_reproduce)
    elif source=='L':
        B01_mult=B01#*water_mask
        write_mult_bands(B01_mult,'B01',size1,size2, geotran,projection_reproduce)
        B02_mult=B02#*water_mask
        write_mult_bands(B02_mult,'B02',size1,size2, geotran,projection_reproduce)
        B03_mult=B03#*water_mask
        write_mult_bands(B03_mult,'B03',size1,size2, geotran,projection_reproduce)
        B04_mult=B04#*water_mask
        write_mult_bands(B04_mult,'B04',size1,size2, geotran,projection_reproduce)
        B8A_mult=B8A#*water_mask
        write_mult_bands(B8A_mult,'B8A',size1,size2, geotran,projection_reproduce)
        B10_mult=B10#*water_mask
        write_mult_bands(B10_mult,'B10',size1,size2, geotran,projection_reproduce)
        B11_mult=B11#*water_mask
        write_mult_bands(B11_mult,'B11',size1,size2, geotran,projection_reproduce)
        B12_mult=B12#*water_mask
        write_mult_bands(B12_mult,'B12',size1,size2, geotran,projection_reproduce)
        B13_mult=B13#*water_mask
        write_mult_bands(B13_mult,'B13',size1,size2, geotran,projection_reproduce)
        B14_mult=B14#*water_mask
        write_mult_bands(B14_mult,'B14',size1,size2, geotran,projection_reproduce)

    water_mask_mult_time += tm.time() - start_time
    logger.debug(f"Ave pred time for {index+1} images: {total_pred_time/(index+1)}")
    
    band_paths_mult = glob.glob(os.path.join(row_outpath, '*1_mult.tif'))
    filename = band_paths_mult[0]
        
    #save RGB image of multiplication
    # Red = B04
    # Green = B03
    # Blue= B02
    unmasked_bgr = np.concatenate([B04, B03, B02], axis=0)
    unmasked_rgb = np.transpose(unmasked_bgr, (1,2,0))
    unmasked_rgb_tmp = np.where(unmasked_rgb==-9999, 0, unmasked_rgb)  # -9999 is the value for pixels that have no data
    unmasked_rgb_min = np.nanmin(unmasked_rgb_tmp)               # get minimum value that isn't one of the missing values
    unmasked_rgb = np.where(unmasked_rgb==-9999, unmasked_rgb_min, unmasked_rgb)    # change "missing" data to minimum so it will just show up as black pixels
    unmasked_rgb = np.nan_to_num(unmasked_rgb)
    unmasked_rgb = (unmasked_rgb-np.nanmin(unmasked_rgb))/(np.nanmax(unmasked_rgb)-np.nanmin(unmasked_rgb))   # Normalize data
    cv2.imwrite(os.path.join(row_outpath, "unmasked_rgb.png"), 255*unmasked_rgb[:,:,::-1])

    bgr = np.concatenate([B04_mult, B03_mult, B02_mult], axis=0)
    rgb = np.transpose(bgr, (1,2,0))    # size: (h,w,3)
    rgb_tmp = np.where(rgb==-9999, 0, rgb)  # -9999 is the value for pixels that have no data
    rgb_min = np.nanmin(rgb_tmp)               # get minimum value that isn't one of the missing values
    rgb = np.where(rgb==-9999, rgb_min, rgb)    # change "missing" data to minimum so it will just show up as black pixels
    rgb = (rgb-np.nanmin(rgb))/(np.nanmax(rgb)-np.nanmin(rgb))   # Normalize data
    rgb = np.nan_to_num(rgb)
    cv2.imwrite(os.path.join(row_outpath, "masked_rgb.png"), 255*rgb[:,:,::-1])


def ssc_preprocessing():


    # do preprocessing

    """
    Created on Feb 15 2023
    Edited through Feb 24 2023

    @author: Luisa Lucchese

    EXTRACT ATTRIBUTES FROM HLS AND SAVE THEM TO A CSV, AND CROP HLS

    Needed: 
        -A HLS file and its path. 
        -A DEM (MERIT DEM suggested, can be adapted to others)
        -The location (lat and long) of the in-situ point from the matchup
        -A python environment as per the specifications given
        -The called function files

    Inputs (obligatory):
        filename - name of HLS with path
        pointlon - longitude of the in-situ point (WGS84)
        pointlat - latitude of the in-situ point (WGS84)
        pathtoDEM - path to the DEM folder
        pathintermediate - path to the folder where intermediate files can be saved
        pathtocsv - path to the folder where csv files can be saved
        pathtocropped - path to the folder where the cropped HLS should be saved
        siteid - the id of the site, to generate unique output names
        
    OBS: in paths, slashes should be like this / and not like this \

    This is a general caller.
    In it, we call some functions that are needed for preprocessing the HLS files
    and extract band values from them.
    For this code, we should provide the arguments such as the name and path
    of the file. If download and matchup should be done, it has to be done before
    running this code. If you need to loop between different instances of the 
    matchup, it should be done in a wrapper that calls this code here.
    """




    row_data = df.iloc[index]
    pointlon = row_data[LON_COL]
    pointlat = row_data[LAT_COL]
    siteid = row_data[SITE_ID_COL]

# =============================================================================
#       moved up
#     band_paths = glob.glob(os.path.join(row_outpath, '*1.tif'))
# 
#     filename = band_paths[0]
# 
# =============================================================================
    # filename = '/media/travis/work/repos/hls_full/aist-hls-data/hls_data_tmp/CBP_WQX-MAT0016/2015-08-17/HLS.L30.T18STH.2015229T154610.v2.0.B01.tif'

    # pathtocsv = os.path.join(row_outpath, 'cropped_imgs')
    
    pathtocropped = os.path.join(row_outpath, 'cropped_imgs')

    if not os.path.exists(pathtocropped):
        os.mkdir(pathtocropped)

    flag_MGRS=0 # if 1, it is to flag that the point is located in a different MGRS 
    # division than the raster tile from HLS. This means that the point is very close
    # to the border of the HLS raster. We cannot ignore this location and not use 
    # this data, but we should be aware that there can be quality concerns about it.


# =============================================================================
#       moved up
#     # should change this parser
#     ncrop=7 #number of characters to crop from the end of the filename, removing 
# 
#     actual_filename, path_to_file, band, tile, date, time, version=extract_info_HLS_filename(filename)
# 
# 
#     # recognize whether it is from Sentinel or Landsat
#     source=recognize_landsat_sentinel(actual_filename)
# =============================================================================

    start_time = tm.time()
    #read DEM and reproject to UTM for sunmask
    filename_w_path_DEM=buildfilename_MERIT_DEM(pointlon,pointlat,DEM_DIR)

    print('dem: '+filename_w_path_DEM)

    epsg_shadow=reproject_to_UTM(pointlat,pointlon,filename_w_path_DEM,DEM,DEM_intermed,8000) #instead of 7680, 8000 (give it a little border)
    #1000m tolerance, generates 1km x 1km DEM cut

    #reproject the HLS file
    reproject_HLS(filename,output_raster_intermed)

    #add the sunmask filter
    year=date[0:4]
    daynum=date[4:7]
    hour=time[0:2]
    minute=time[2:4]
    second=time[4:7]

    sunmask=function_sunmask(DEM,pointlon,pointlat,year,daynum,hour,minute,second,dst_filepath)

    # output_raster_warped -- save the path to shadowproj1
    sunmask_as_array,output_raster_warped=reproject_sunmask(dst_filepath,INTERMED_OUT_DIR,output_raster_intermed,epsg_shadow,source)#
    print('sunmask_as_array: '+str(np.nansum(sunmask_as_array)))
    dem_shadow_time += tm.time()-start_time

    start_time = tm.time()
    #read the Fmask to mask based on cloud, cloud shadow, and snow.
    try:
        fmask=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'Fmask')
        print('Fmask file: '+fmask)
        fmask = fmask.replace('_mult', '') #this is important because we need the mask untouched for the binary operations
        reproject_HLS(fmask,output_raster_intermed)
    except:
        fmask=build_HLS_filenames_extrabands_namepres(filename,'Fmask',ncrop)  
        print('Attention: point outside MGRS bounds for the tile. Fmask file: '+fmask)
        fmask = fmask.replace('_mult', '')
        reproject_HLS(fmask,output_raster_intermed)
        flag_MGRS=1


    mask_Fmask=read_Fmask(output_raster_intermed, maskedout)
    fmask_shadow_time += tm.time() - start_time

    start_time = tm.time()
    #regular band loop
    nbands=14 #12 regular bands to loop, +2 when nomenclature changed
    nout=6 #number of outputs in the extract_value function
    reading_wos=np.zeros((nbands+6)*nout+2) #this is the reading without sunmask
    reading_sun=np.zeros((nbands+6)*nout+2) #this is the reading with sunmask
    bandread=0 #initializing
    #if source =='L': #Landsat - we no longer use those because we adapted band names
    if flag_MGRS==0:
        for iband in range(0,(nbands)*nout,nout): #bands 1 to 11
            bandread=bandread+1
            try:
                filename_extract=build_HLS_filenames(date,time,pointlon,pointlat,path_to_file,bandread, source)
                reproject_HLS(filename_extract,output_raster_intermed)
                # 
                reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
                # 
            except:
                print('Could not read HLS, band number: ', str(bandread))
                filler=np.zeros(nout)
                filler[:]=np.nan
                reading_wos[iband:(iband+nout)]=filler
                reading_sun[iband:(iband+nout)]=filler
    else:
        for iband in range(0,nbands*nout,nout):
            bandread=bandread+1
            try:
                bandnum_str='B'+str(bandread).zfill(2)
                filename_extract=build_HLS_filenames_extrabands_namepres(filename,bandnum_str,ncrop)
                reproject_HLS(filename_extract,output_raster_intermed)
                reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
                #
            except Exception as e:
                print(e)
                print('flag_MGRS ON. Could not read HLS, band number: ', str(bandread))
                filler=np.zeros(nout)
                filler[:]=np.nan
                reading_wos[iband:(iband+nout)]=filler
                reading_sun[iband:(iband+nout)]=filler
    

    
#save 8A band
    if flag_MGRS==0:
        iband=iband+nout
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'B8A')
        reproject_HLS(filename_extract,output_raster_intermed)
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        
    else:
        iband=iband+nout
        filename_extract=build_HLS_filenames_extrabands_namepres(filename,'B8A',ncrop)#(date,time,pointlon,pointlat,path_to_file,source,'B8A')
        reproject_HLS(filename_extract,output_raster_intermed)
        #
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        

    #saving mask bands
    band_names = ['SAA', 'SZA', 'VZA', 'VAA', 'Fmask']
    for i in band_names:
        Band = rasterio.open(os.path.join(image_path+i+'.tif')).read()
        Band_mult=Band#*water_mask
        write_mult_bands(Band_mult,i,size1,size2, geotran,projection_reproduce)
    
    if flag_MGRS==0:
        #Fmask band min, max, and average
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'Fmask')
        reproject_HLS(filename_extract,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #SAA band min, max, and average
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'SAA')
        iband=iband+nout
        reproject_HLS(filename_extract,output_raster_intermed)
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
    
        
        #SZA band min, max, and average
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'SZA')
        iband=iband+nout
        reproject_HLS(filename_extract,output_raster_intermed)
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #VAA band min, max, and average
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'VAA')
        iband=iband+nout
        reproject_HLS(filename_extract,output_raster_intermed)
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #

        #VZA band min, max, and average
        filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'VZA')
        iband=iband+nout
        reproject_HLS(filename_extract,output_raster_intermed)
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #

    else:
        #flag_MGRS=1 #flagging MGRS concerns.
        print('Attention: point outside MGRS bounds for the tile.')
        
        #fmask
        fmask=build_HLS_filenames_extrabands_namepres(filename,'Fmask',ncrop)  
        reproject_HLS(fmask,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #SAA
        filename_extract=build_HLS_filenames_extrabands_namepres(filename,'SAA',ncrop)  
        reproject_HLS(filename_extract,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #SZA
        filename_extract=build_HLS_filenames_extrabands_namepres(filename,'SZA',ncrop)  
        reproject_HLS(filename_extract,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #VAA
        filename_extract=build_HLS_filenames_extrabands_namepres(filename,'VAA',ncrop)  
        reproject_HLS(filename_extract,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
        #VZA
        filename_extract=build_HLS_filenames_extrabands_namepres(filename,'VZA',ncrop)  
        reproject_HLS(filename_extract,output_raster_intermed)
        iband=iband+nout
        reading_wos[iband:(iband+nout)], reading_sun[iband:(iband+nout)]=extract_value(filename_extract,pointlon,pointlat,sunmask_as_array,output_raster_intermed, mask_Fmask, dst_filename_wos,dst_filename_sun)
        #
        
    #save the MGRS flag as well.
    iband=iband+nout

    reading_wos[iband]=flag_MGRS
    reading_sun[iband]=flag_MGRS

   
    feat_extract_time += tm.time() - start_time

    #save the landsat or sentinel flag too
    iband=iband+1
    if source=='L':
        reading_wos[iband]=0 
        reading_sun[iband]=0 
    elif source=='S':
        reading_wos[iband]=1
        reading_sun[iband]=1 
    else:
        reading_wos[iband]=2 
        reading_sun[iband]=2
    
    #saving it all into a csv file

    start_time = tm.time()
 #first, without the sunmask
    makecsvname_wos= os.path.join(row_outpath,'WOS_'+siteid+'_'+date+'.csv')#actual_filename[0:-15]+'_'+unique_str+'.csv'

    columns = ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
               'b2_median','b2_count','b3_min','b3_mean','b3_max','b3_std','b3_median','b3_count','b4_min','b4_mean','b4_max','b4_std',\
                'b4_median','b4_count','b5_min','b5_mean','b5_max','b5_std','b5_median','b5_count','b6_min','b6_mean','b6_max','b6_std',\
                    'b6_median','b6_count','b7_min','b7_mean','b7_max','b7_std','b7_median','b7_count','b8_min','b8_mean','b8_max','b8_std',\
                        'b8_median','b8_count','b9_min','b9_mean','b9_max','b9_std','b9_median','b9_count','b10_min','b10_mean','b10_max','b10_std',\
                            'b10_median','b10_count','b11_min','b11_mean','b11_max','b11_std','b11_median','b11_count','b12_min','b12_mean','b12_max','b12_std',\
                                'b12_median','b12_count','b13_min','b13_mean','b13_max','b13_std',\
                                    'b13_median','b13_count','b14_min','b14_mean','b14_max','b14_std',\
                                        'b14_median','b14_count','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','b8a_count','Fmask_min','Fmask_mean','Fmask_max',\
                                            'Fmask_std','Fmask_median','Fmask_count','SAA_min','SAA_mean','SAA_max','SAA_std','SAA_median','SAA_count','SZA_min','SZA_mean',\
                                                'SZA_max','SZA_std','SZA_median','SZA_count','VAA_min','VAA_mean','VAA_max','VAA_std','VAA_median','VAA_count','VZA_min','VZA_mean',\
                                                    'VZA_max','VZA_std','VZA_median','VZA_count','MGRS','LorS']

    reading_wos = np.asarray(reading_wos)
    out_df = pd.DataFrame(reading_wos).T
    out_df.columns = columns
    row = pd.DataFrame(df.iloc[index]).T.reset_index()
    result = out_df

    result['SiteID'] = row['SiteID']
    result['lat'] = row['lat']
    result['lon'] = row['lon']
    result['date'] = row['date']
    result['cloud_cover'] = row['cloud_cover']
    result['tss_value'] = row['tss_value']
    result['relative_day']  = row['relative_day']
    # result = result.loc[:, result.columns.str.contains('^Unnamed')]
    print('WOS Result', result)
    result.to_csv(makecsvname_wos)
    save_to_csv_time += tm.time() - start_time
    ########################## turn this into csv, add in all the measurements from the input row ####################################

    #second, with the sunmask
    makecsvname_sun= os.path.join(row_outpath,'SUN_'+siteid+'_'+date+'.csv')#actual_filename[0:-15]+'_'+unique_str+'.csv'

    columns = ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
               'b2_median','b2_count','b3_min','b3_mean','b3_max','b3_std','b3_median','b3_count','b4_min','b4_mean','b4_max','b4_std',\
                'b4_median','b4_count','b5_min','b5_mean','b5_max','b5_std','b5_median','b5_count','b6_min','b6_mean','b6_max','b6_std',\
                    'b6_median','b6_count','b7_min','b7_mean','b7_max','b7_std','b7_median','b7_count','b8_min','b8_mean','b8_max','b8_std',\
                        'b8_median','b8_count','b9_min','b9_mean','b9_max','b9_std','b9_median','b9_count','b10_min','b10_mean','b10_max','b10_std',\
                            'b10_median','b10_count','b11_min','b11_mean','b11_max','b11_std','b11_median','b11_count','b12_min','b12_mean','b12_max','b12_std',\
                                'b12_median','b12_count','b13_min','b13_mean','b13_max','b13_std',\
                                    'b13_median','b13_count','b14_min','b14_mean','b14_max','b14_std',\
                                        'b14_median','b14_count','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','b8a_count','Fmask_min','Fmask_mean','Fmask_max',\
                                            'Fmask_std','Fmask_median','Fmask_count','SAA_min','SAA_mean','SAA_max','SAA_std','SAA_median','SAA_count','SZA_min','SZA_mean',\
                                                'SZA_max','SZA_std','SZA_median','SZA_count','VAA_min','VAA_mean','VAA_max','VAA_std','VAA_median','VAA_count','VZA_min','VZA_mean',\
                                                    'VZA_max','VZA_std','VZA_median','VZA_count','MGRS','LorS']

    reading_sun = np.asarray(reading_sun)
    out_df = pd.DataFrame(reading_sun).T
    out_df.columns = columns
    row = pd.DataFrame(df.iloc[index]).T.reset_index()

    result = out_df

    row = df.iloc[index]
    result['SiteID'] = row['SiteID']
    result['lat'] = row['lat']
    result['lon'] = row['lon']
    result['date'] = row['date']
    result['cloud_cover'] = row['cloud_cover']
    result['tss_value'] = row['tss_value']
    result['relative_day']  = row['relative_day']
    # result = result.loc[:, result.columns.str.contains('^Unnamed')]
    print('SUN Result', result)

    # result.reset_index(inplace=True)
    result.to_csv(makecsvname_sun)



    # end of preprocessing
    start_time = tm.time()
    #now, save a cropped version of the image for Rangel (CV purposes)
    buffersize=7680
    if flag_MGRS==0:
        crop_HLS_save(filename,pointlon,pointlat,siteid,pathtocropped,buffersize,source,nbands,output_raster_warped)
    else:
        crop_HLS_save_flagMGRS1(filename,pointlon,pointlat,siteid,pathtocropped,buffersize,ncrop,source,nbands,output_raster_warped)
    
    save_cropped_imgs_time += tm.time() - start_time

    # remove non-cropped images
    start_time = tm.time()
    remove_all_non_cropped_imgs(row_outpath=row_outpath)
    remove_large_imgs_time += tm.time() - start_time

    logger.debug(f"dl_time: {dl_time}s")
    logger.debug(f"img_loading_time: {img_loading_time}s")
    logger.debug(f"water_mask_time: {water_mask_time}s")
    logger.debug(f"total_pred_time: {total_pred_time}s")
    logger.debug(f"water_mask_mult_time: {water_mask_mult_time}s")
    logger.debug(f"dem_shadow_time: {dem_shadow_time}s")
    logger.debug(f"fmask_shadow_time: {fmask_shadow_time}s")
    logger.debug(f"feat_extract_time: {feat_extract_time}s")
    logger.debug(f"save_to_csv_time: {save_to_csv_time}s")
    logger.debug(f"save_cropped_imgs_time: {save_cropped_imgs_time}s")
    logger.debug(f"remove_large_imgs_time: {remove_large_imgs_time}s")
    logger.debug(f"overall time: {tm.time() - overall_start_time}s")
    
