# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:57:07 2025

Run just SSC - for small sets
Runs after preprocessing

@author: Luisa Lucchese
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import pickle
import os
import sys
import time
from .auxfunctions.f_execens import f_execens
import logging





# Clear all previously registered custom objects
tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}

@tf.keras.utils.register_keras_serializable(package="my_package", name="loss_function_m1")
def loss_function_m1(y_true, y_pred):
   percentageerror = tf.divide(tf.square(y_pred- y_true),y_true+0.01)
   return tf.reduce_mean(percentageerror, axis=-1)

def list_files_in_folder(folder_path):
    # List to store file names
    files_list = []
    
    # Walk through folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            files_list.append(os.path.join(root, file))
    
    return files_list

def ann_ssc_model(df_hlsprocessed_raw, model_dir):
    
    # Start the timer
    start_time = time.time()

    #load models

    path_load_m1= os.path.join(model_dir, "gl_20250522_2_m1")
    path_load_m2= os.path.join(model_dir, "gl_20250522_2_m2")

    model1_load = tf.keras.models.load_model(path_load_m1)

    model2_load = tf.keras.models.load_model(path_load_m2)

    #loading variables
    with open(path_load_m1+'/maxval_cut.pkl', 'rb') as file:
        maxval_cut = pickle.load(file)    
    with open(path_load_m1+'/maximum_out.pkl', 'rb') as file:
        maximum_out = pickle.load(file)     
    with open(path_load_m1+'/max_m2.pkl', 'rb') as file:
        max_m2 = pickle.load(file)     
    with open(path_load_m1+'/m2_x_train.pkl', 'rb') as file:
        m2_x_train = pickle.load(file)    
    with open(path_load_m1+'/min_input_m1.pkl', 'rb') as file:
        min_input_m1 = pickle.load(file) 
    with open(path_load_m1+'/max_input_m1.pkl', 'rb') as file:
        max_input_m1 = pickle.load(file) 
    with open(path_load_m1+'/min_input_m2.pkl', 'rb') as file:
        min_input_m2 = pickle.load(file) 
    with open(path_load_m1+'/max_input_m2.pkl', 'rb') as file:
        max_input_m2 = pickle.load(file) 


    #load data



    #df_merged_2=pd.read_csv("D:\Luisa\initial_tests_SSC_preproc\ssc_out_nov_29_final\sun_test_set.csv")#("D:\Luisa\initial_tests_SSC_preproc\ssc_out_nov_06\sun_test_set.csv")
    #("D:/Luisa/initial_tests_SSC_preproc/ssc_out_sept_27/drive-download-20230927T142903Z-001/wos_test_set.csv")
    #pd.read_csv("D:\Luisa\data\datasamples_11k\sun_test_set.csv")

    #path_folder='D:/Luisa/data/jan_02_2025_europe/preprocessed/results_jan_9_2025/results'#"D:/Luisa/data/oct_17_preproc/results_one_fail/results/" #oct_11_prelim_preproc/results/"#"D:/Luisa/data/all_nodes_ssc_preprocessing_may_16/results/"
    #path_folder='D:/Luisa/data/deployment_htc_2025_feb/NSF_points_HLSs/origpoint/preproc'
    # path_folder='D:/Luisa/data/in_situ_Punwath_2025/processed/csv_out'#'D:/Luisa/data/globalrun_2025_03_11/test_dataset_for_Travis/whole_image'#'D:/Luisa/data/deployment_htc_2025_feb/NSF_points_HLSs/USGS_station_pt/preproc'

    # path_save_results='D:/Luisa/data/in_situ_Punwath_2025/processed/ANNrun/'#'D:/Luisa/data/globalrun_2025_03_11/test_dataset_for_Travis/whole_image/ANNrun/'#'D:/Luisa/data/deployment_htc_2025_feb/NSF_points_HLSs/USGS_station_pt/ANNrun/'
    # #"D:/Luisa/data/jan_02_2025_europe/ANNrun/" #oct_11_prelim_preproc/ANNrun/"#"D:/Luisa/data/all_nodes_ssc_preprocessing_may_16/ANN_run/"

    # path_csv_reaches="D:/Luisa/data/jan_02_2025_europe/coordinates_csv/all_reaches_merged.csv"

    # coords_reach = pd.read_csv(path_csv_reaches)

    # df_stacked=pd.DataFrame()

    # files_path = list_files_in_folder(path_folder)
    # y_mod_and_coord_avg_all=np.zeros([0,7]) #6 is the number of things saved
    # saveptind=0
    # for file_path in files_path:
        
    #     df_hlsprocessed_raw = pd.read_csv(file_path)
        
    #     file_name = os.path.basename(file_path)
    if "S" in df_hlsprocessed_raw['LorS'].unique():
        # print(f"Pattern 'S30' found in file: {file_name}")
        df_hlsprocessed = df_hlsprocessed_raw.rename(columns=lambda x: x.replace('B05', 'NA1') if 'B05' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B8A', 'B05') if 'B8A' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B06', 'NA2') if 'B06' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B11', 'B06') if 'B11' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B07', 'NA3') if 'B07' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B12', 'B07') if 'B12' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B09', 'NA4') if 'B09' in x else x)
        df_hlsprocessed = df_hlsprocessed.rename(columns=lambda x: x.replace('B10', 'B09') if 'B10' in x else x)

    elif "L" in df_hlsprocessed_raw['LorS'].unique():
        # Landsat, doesnt need band name adaptations
        df_hlsprocessed=df_hlsprocessed_raw
   
    
    mapping = {'L': 0, 'S': 1}
    df_hlsprocessed['LorS'] = df_hlsprocessed['LorS'].replace(mapping)

    logging.info("after mappig", df_hlsprocessed['LorS'])
    
    # empty output column
    df_hlsprocessed['SSC'] = 0
    
    #dataframe_array=df_hlsprocessed.to_numpy()
    
    
    # input_cols= ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
    #                 'b2_median','b3_min','b3_mean','b3_max','b3_std','b3_median','b4_min','b4_mean','b4_max','b4_std',\
    #                 'b4_median','b10_min','b10_mean','b10_max','b10_std',\
    #                             'b10_median','b11_min','b11_mean','b11_max','b11_std','b11_median','b12_min','b12_mean','b12_max','b12_std',\
    #                                 'b12_median','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','LorS']
    
    # input_cols= ['B01_min','B01_mean','B01_max','B01_std','B01_median','B01_count','B02_min','B02_mean','B02_max','B02_std',\
    #                 'B02_median','B03_min','B03_mean','B03_max','B03_std','B03_median','B04_min','B04_mean','B04_max','B04_std',\
    #                 'B04_median','B09_min','B09_mean','B09_max','B09_std',\
    #                             'B09_median','B06_min','B06_mean','B06_max','B06_std','B06_median','B07_min','B07_mean','B07_max','B07_std',\
    #                                 'B07_median','B05_min','B05_mean','B05_max','B05_std','B05_median','LorS']
    input_cols = [
    'B01_min','B01_mean','B01_max','B01_std','B01_median','B01_wmean','B01_wstd','B01_wmedian','B01_count',
    'B02_min','B02_mean','B02_max','B02_std','B02_median','B02_wmean','B02_wstd','B02_wmedian',
    'B03_min','B03_mean','B03_max','B03_std','B03_median','B03_wmean','B03_wstd','B03_wmedian',
    'B04_min','B04_mean','B04_max','B04_std','B04_median','B04_wmean','B04_wstd','B04_wmedian',
    'B09_min','B09_mean','B09_max','B09_std','B09_median','B09_wmean','B09_wstd','B09_wmedian',
    'B06_min','B06_mean','B06_max','B06_std','B06_median','B06_wmean','B06_wstd','B06_wmedian',
    'B07_min','B07_mean','B07_max','B07_std','B07_median','B07_wmean','B07_wstd','B07_wmedian',
    'B05_min','B05_mean','B05_max','B05_std','B05_median','B05_wmean','B05_wstd','B05_wmedian',
    'LorS'
    ]

    
    output_cols=['SSC']#['tss_value']
    
    coords_cols= ['lat','lon']
    
    #limits for m1
    minimum_out=-1.0 #in log scale 
    
    varnum=len(input_cols)#
    
    min_m2=maxval_cut*maximum_out
    
    #function definition
    def clamp_vector(n,minn, maxn,flag):
        shp1=len(n)
        flag=np.zeros_like(n)
        shp2=1
        for i in range(0,shp1):
            for j in range(0,shp2):
                if n[i] < minn:
                    n[i]= minn
                    flag[i]=-1
                elif n[i]  > maxn:
                    n[i]= maxn
                    flag[i]=1
        return n,flag
    
    
    
    data_input=df_hlsprocessed[input_cols]
    data_input_array=data_input.to_numpy()
    
    data_input_array_noinf=data_input_array
    #remove samples with infinite values 
    data_input_array_noinf[data_input_array_noinf == float('+inf')]=float('nan')
    
    # travis isnan not supported bugfix
    data_input_array_noinf = data_input_array_noinf.astype(float)
    
    #filter input
    locations_nan_input=np.isnan(data_input_array_noinf).any(axis=1)
    
    data_input_array_filtered=data_input_array[~locations_nan_input]
    
    data_output=df_hlsprocessed[output_cols]
    data_output_array=data_output.to_numpy()
    
    data_output_array_filtered=data_output_array[~locations_nan_input]
    
    #filter output
    locations_nan_output=np.isnan(data_output_array_filtered).any(axis=1)
    
    data_input_array_filtered2=data_input_array_filtered[~locations_nan_output]
    data_output_array_filtered2=data_output_array_filtered[~locations_nan_output]
    
    data_coords=df_hlsprocessed[coords_cols]
    data_coords_array=data_coords.to_numpy()
    data_coords_array_filtered=data_coords_array[~locations_nan_input]
    data_coords_array_filtered2=data_coords_array_filtered[~locations_nan_output]
    
    if not data_coords_array_filtered2.size >0:
        print('all nans found in preprocessing')
        # sys.exit(0)
        raise ValueError('all nans found in preprocessing')
    else:
    
        y_modeled,y_observed,coords_obs= f_execens(model1_load,model2_load,data_input_array_filtered2,data_output_array_filtered2,data_coords_array_filtered2,minimum_out,maximum_out,min_m2,max_m2,min_input_m1,max_input_m1,min_input_m2,max_input_m2,maxval_cut,varnum) #
    
        
        #y_modeled=np.append(predicted_m2_scaled,predicted_m1_scaled)
        
        #no concentration value below zero
        y_modeled_zero = y_modeled.copy()
        y_modeled_zero[y_modeled_zero < 0] = 0
        
        # find reach id for given coordinates
        # Truncate function for lat/lon to three decimal places
        def truncate_to_three_decimal_places(value):
            return np.floor(value * 1000) / 1000
        # Truncate coords_obs to three decimal places
        coords_obs_truncated = np.array([
            [truncate_to_three_decimal_places(lat), truncate_to_three_decimal_places(lon)] for lat, lon in coords_obs
            ])
        # Truncate coords_reach dataframe's Latitude and Longitude columns to three decimal places
        df_hlsprocessed['Latitude_trunc'] = df_hlsprocessed['lat'].apply(truncate_to_three_decimal_places)
        df_hlsprocessed['Longitude_trunc'] = df_hlsprocessed['lon'].apply(truncate_to_three_decimal_places)
        coords_reach_unique = df_hlsprocessed.drop_duplicates(subset=['Latitude_trunc', 'Longitude_trunc'])
        # Create a df from truncated coords_obs
        df_coords_obs = pd.DataFrame(coords_obs_truncated, columns=['Latitude_trunc', 'Longitude_trunc'])
        df_coords_obs['SSC'] = y_modeled_zero
        # df_coords_obs['date']=df_hlsprocessed_raw['date']
        # Merge coords_obs with coords_reach on the truncated coordinates
        merged_data_coords = pd.merge(df_coords_obs, coords_reach_unique, on=['Latitude_trunc', 'Longitude_trunc'], how='left')
        # Extract the Discharge (reach IDs) for the matched coordinates
        # matched_reach_ids = merged_data_coords['ReachID'].values ------------ this was in

        #desired_order = ['Latitude', 'Longitude', 'SSC','date','flanking','ReachID']
        df_reordered = merged_data_coords #df_merged_ann_dis[desired_order]
        
        
        #save to csv file
        # makecsvname=(path_save_results+'/'+'ssc_only_'+os.path.basename(file_path)+'.csv')#
        # df_reordered.to_csv(makecsvname)

        #grouped_mean = df_reordered.groupby(['ReachID', 'date', 'flanking'], as_index=False).mean()
        #df_stacked = pd.concat([df_stacked, grouped_mean], ignore_index=True)

        
    #save to csv file
    #makecsvname=(path_save_results+'/'+'meancoords_SSConly'+'.csv')#
    #df_stacked.to_csv(makecsvname)

    # Clean df

    # Rename 'SSC_x' to 'SSC' if it exists
    if 'SSC_x' in df_reordered.columns:
        df_reordered.rename(columns={'SSC_x': 'SSC'}, inplace=True)

    # Drop columns if they exist
    columns_to_drop = ['SSC_y', 'Latitude_trunc', 'Longitude_trunc']
    df_reordered.drop(columns=[col for col in columns_to_drop if col in df_reordered.columns], inplace=True)


    mapping = {0:'L', 1:'S'}
    df_reordered['LorS'] = df_reordered['LorS'].replace(mapping)

    # End the timer
    end_time = time.time()

    # Calculate the dispensed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return df_reordered
    # =============================================================================
    # columns = ['lat', 'lon', 'SSC_modeled', 'discharge_SWOT', 'sedflux','date','ReachID']
    # rownumber,foo=y_mod_and_coord_avg_all.shape
    # with open(makecsvname, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(columns)
    #     for row in range(0,rownumber):
    #         row_to_write=y_mod_and_coord_avg_all[row]
    #         writer.writerow(row_to_write)     
    # =============================================================================
    
        

