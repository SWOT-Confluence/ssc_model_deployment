# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:56:54 2023

@author: Luisa Lucchese

Execute a model composed of m1 and m2 in certain samples (points)
and save the output.
Made for deployment.
"""

# Standard imports
import pickle
import os
import csv

# Third-party imports
import pandas as pd
import numpy as np
import tensorflow as tf

# Local imports

# Functions
def clamp_vector(n,minn, maxn,flag):
    """
    Function definition, needed for flagging samples for M2
    """
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





    # path_load_m1='/opt/models/model_static_files/INC_20231117_2_m1'
    # path_load_m2='/opt/models/model_static_files/INC_20231117_2_m2'

def ann_ssc_model(preprocessed_dataframe, ann_ssc_model_outpath, m1_path, m2_path, predict_both_models_bool, predict_for_test_bool=False):
    
    """
    Setup Model
    """

    # Set true if you want to use both m1 and m2 in the range that they are both valid.
    predict_both_models=predict_both_models_bool

    # Set true if you want to use the models to predict for the test set. It is useful for testing purposes.
    predict_for_test=predict_for_test_bool 

    #the features used in the models. Alter accordingly.
    input_cols= ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
                    'b2_median','b3_min','b3_mean','b3_max','b3_std','b3_median','b4_min','b4_mean','b4_max','b4_std',\
                    'b4_median','b10_min','b10_mean','b10_max','b10_std',\
                                'b10_median','b11_min','b11_mean','b11_max','b11_std','b11_median','b12_min','b12_mean','b12_max','b12_std',\
                                    'b12_median','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','LorS']

    # load models
    path_load_m1 = m1_path
    path_load_m2=m2_path

    model1_load = tf.keras.models.load_model(path_load_m1)
    model2_load = tf.keras.models.load_model(path_load_m2)

    #load variables   
    with open(path_load_m1+'/x_test.pkl', 'rb') as file:
        x_test = pickle.load(file)   
    with open(path_load_m1+'/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file) 
    with open(path_load_m1+'/maxval_cut.pkl', 'rb') as file:
        maxval_cut = pickle.load(file)    
    with open(path_load_m1+'/maximum_out.pkl', 'rb') as file:
        maximum_out = pickle.load(file)     
    with open(path_load_m1+'/max_m2.pkl', 'rb') as file:
        max_m2 = pickle.load(file)      
    with open(path_load_m1+'/min_input_m1.pkl', 'rb') as file:
        min_input_m1 = pickle.load(file)     
    with open(path_load_m1+'/max_input_m1.pkl', 'rb') as file:
        max_input_m1 = pickle.load(file)    
    with open(path_load_m1+'/min_input_m2.pkl', 'rb') as file:
        min_input_m2 = pickle.load(file)     
    with open(path_load_m1+'/max_input_m2.pkl', 'rb') as file:
        max_input_m2 = pickle.load(file) 


    """
    Load data to predict on
    """

    # Load preprocessed target data
    dataframe_array=preprocessed_dataframe.to_numpy()

    #Define outfile of model (.csv)
    makecsvname=(ann_ssc_model_outpath)


    """
    Filter data
    """
    output_cols=['tss_value'] #the output, when available, for testing purposes

    coords_cols= ['lat','lon'] #useful for map making

    # filtering options
    min_m1=0 #no concentration below 0 is possible
    max_m1=maximum_out
    varnum=len(input_cols) #number of variables
    min_m2=maxval_cut*maximum_out

    #filter the data - take out invalid values
    data_input=preprocessed_dataframe[input_cols]
    data_input_array=data_input.to_numpy()
    data_input_array_noinf=data_input_array

    #remove samples with infinite values 
    data_input_array_noinf[data_input_array_noinf == float('+inf')]=float('nan')

    #filter input
    locations_nan_input=np.isnan(data_input_array_noinf).any(axis=1)
    data_input_array_filtered=data_input_array[~locations_nan_input]
    data_output=preprocessed_dataframe[output_cols]
    data_output_array=data_output.to_numpy()
    data_output_array_filtered=data_output_array[~locations_nan_input]

    #filter output
    locations_nan_output=np.isnan(data_output_array_filtered).any(axis=1)
    data_input_array_filtered2=data_input_array_filtered[~locations_nan_output]
    data_output_array_filtered2=data_output_array_filtered[~locations_nan_output]
    data_coords=preprocessed_dataframe[coords_cols]
    data_coords_array=data_coords.to_numpy()
    data_coords_array_filtered=data_coords_array[~locations_nan_input]
    data_coords_array_filtered2=data_coords_array_filtered[~locations_nan_output]

    #normalize
    #for the inputs, normalize based on maximum and minimum of the training set
    x_norm=(data_input_array_filtered2-min_input_m1)/(max_input_m1-min_input_m1)

    #predict based on m1
    predicted_m1=model1_load.predict(x_norm)

    #flag samples for m1 or intersection (later, we flag for m2)
    y_norm_eval,y_norm_eval_flag=clamp_vector(predicted_m1,0,maxval_cut,np.zeros_like(predicted_m1))

    #values were flagged,  
    #now let's select the registers for which the high concentration
    #model should be run
    [size1, size2]=y_norm_eval_flag.shape

    m1_x_eval=np.zeros([0,varnum])
    m1_y_eval=np.zeros(0)
    m1_data_coords=np.zeros([0,2])
    for i in range(0,size1):
        if y_norm_eval_flag[i,0]==0: #only m1
            temporary_x_eval=data_input_array_filtered2[i,0:varnum]
            m1_x_eval=np.row_stack((m1_x_eval,temporary_x_eval))#
            m1_y_eval = np.append(m1_y_eval, data_output_array_filtered2[i, 0])
            #drag coordinate info
            m1_data_coords=np.row_stack((m1_data_coords,data_coords_array_filtered2[i, :]))
    if m1_x_eval.size > 0:
        y_eval_m1=(m1_y_eval-min_m1)/(max_m1-min_m1)
        x_eval_m1=(m1_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
        predicted_m1=model1_load.predict(x_eval_m1)
        predicted_m1_scaled=predicted_m1*(max_m1-min_m1)+min_m1
    else:
        predicted_m1_scaled=[]   


    mb_x_eval=np.zeros([0,varnum])
    mb_y_eval=np.zeros(0)
    mb_data_coords=np.zeros([0,2])
    mi_x_eval=np.zeros([0,varnum])
    mi_y_eval=np.zeros(0)
    mi_data_coords=np.zeros([0,2])
    m2_x_eval=np.zeros([0,varnum])
    m2_y_eval=np.zeros(0)
    m2_data_coords=np.zeros([0,2])
    index_flagm2=np.empty((size1))
    index_flagm2[:] = np.nan
    counter=0
    for i in range(0,size1):
        if y_norm_eval_flag[i,0]==1: #intersection, or m2 (needs to be defined again)
            temporary_x_eval=data_input_array_filtered2[i,0:varnum]
            mb_x_eval=np.row_stack((mb_x_eval,temporary_x_eval))#
            mb_y_eval = np.append(mb_y_eval, data_output_array_filtered2[i, 0])
            #drag coordinate info
            mb_data_coords=np.row_stack((mb_data_coords,data_coords_array_filtered2[i, :]))
            #we need to also save the index for compatibility
            index_flagm2[i]=counter
            counter=counter+1
    if mb_x_eval.size > 0:
        y_eval_mb2=(mb_y_eval-min_m2)/(max_m2-min_m2)
        x_eval_mb2=(mb_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#5
        predicted_mb2=model2_load.predict(x_eval_mb2)
        predicted_mb2_scaled=predicted_mb2*(max_m2-min_m2)+min_m2
        #flag samples for m2
        foo,y_norm_eval_flag_m2=clamp_vector(predicted_mb2,max_m1,max_m2,np.zeros_like(predicted_mb2))
        for i in range(0,size1):
            #important, indexes compatibility 
            i_flagm2=index_flagm2[i]
            if np.isnan(i_flagm2)==False:
                i_flagm2_int=int(i_flagm2)
                if y_norm_eval_flag_m2[i_flagm2_int,0]==-1: #real intersection
                    temporary_x_eval=data_input_array_filtered2[i,0:varnum]
                    mi_x_eval=np.row_stack((mi_x_eval,temporary_x_eval))#
                    mi_y_eval = np.append(mi_y_eval, data_output_array_filtered2[i, 0])
                    #drag coordinate info
                    mi_data_coords=np.row_stack((mi_data_coords,data_coords_array_filtered2[i, :]))
        if mi_x_eval.size > 0:
                #run m1
                y_eval_mi1=(mi_y_eval-min_m1)/(max_m1-min_m1)
                x_eval_mi1=(mi_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
                predicted_mi1=model1_load.predict(x_eval_mi1)
                predicted_mi1_scaled=predicted_mi1*(max_m1-min_m1)+min_m1
                #run m2
                y_eval_mi2=(mi_y_eval-min_m2)/(max_m2-min_m2)
                x_eval_mi2=(mi_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
                predicted_mi2=model2_load.predict(x_eval_mi2)
                predicted_mi2_scaled=predicted_mi2*(max_m2-min_m2)+min_m2
                if predict_both_models:
                    #use both models
                    predicted_mi_scaled=(predicted_mi1_scaled+predicted_mi2_scaled)/2.0 
                else:
                    predicted_mi_scaled=predicted_mi2_scaled #use just m2
        else:
            predicted_mi_scaled=[] 
            
    for i in range(0,size1):
        #important, indexes compatibility 
        i_flagm2=index_flagm2[i]
        if np.isnan(i_flagm2)==False:
            i_flagm2_int=int(i_flagm2)
            if y_norm_eval_flag_m2[i_flagm2_int,0]>-1: #only m2
                temporary_x_eval=data_input_array_filtered2[i,0:varnum]
                m2_x_eval=np.row_stack((m2_x_eval,temporary_x_eval))#
                m2_y_eval = np.append(m2_y_eval, data_output_array_filtered2[i, 0])
                #drag coordinate info
                m2_data_coords=np.row_stack((m2_data_coords,data_coords_array_filtered2[i, :]))
    if m2_x_eval.size > 0:
        y_eval_m2=(m2_y_eval-min_m2)/(max_m2-min_m2)
        x_eval_m2=(m2_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
        predicted_m2=model2_load.predict(x_eval_m2)
        predicted_m2_scaled=predicted_m2*(max_m2-min_m2)+min_m2
    else:
        predicted_m2_scaled=[]

    y_observed_step1=np.append(m1_y_eval,mi_y_eval)
    y_observed=np.append(y_observed_step1,m2_y_eval)

    coords_obs_step1=np.row_stack((m1_data_coords,mi_data_coords))
    coords_obs=np.row_stack((coords_obs_step1,m2_data_coords))

    y_modeled_step1=np.append(predicted_m1_scaled,predicted_mi_scaled)        
    y_modeled_log=np.append(y_modeled_step1,predicted_m2_scaled)      

    #bringing y_modeled back from log scale
    y_modeled=np.exp(y_modeled_log)
    
    #no concentration value below zero
    y_modeled_zero = y_modeled.copy()
    y_modeled_zero[y_modeled_zero < 0] = 0

    #to make maps
    y_mod_and_coord=np.column_stack((coords_obs,y_modeled_zero))
    y_mod_and_coord_calc=np.column_stack((y_mod_and_coord,y_observed,y_modeled_zero-y_observed))
    #

    #save to csv file

    columns = ['lat', 'lon', 'SSC_modeled']
    rownumber,foo=y_mod_and_coord.shape
    with open(makecsvname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in range(0,rownumber):
            row_to_write=y_mod_and_coord[row]
            writer.writerow(row_to_write)


    #statistics for all of the csv given. Attention, can mix train, val, and test

    rmse_2models_all=np.sqrt(np.sum(np.absolute(y_modeled_zero.flatten()-y_observed.flatten()))**2.0)/len(y_observed.flatten())

    mae_all=sum(abs(y_modeled_zero-y_observed))/len(y_modeled_zero)

    max_err_all=max(abs(y_modeled_zero-y_observed))

    abs_err_all=abs(y_modeled_zero-y_observed)
    #quartiles of error
    quartiles_err_1_all=np.quantile(abs_err_all,0.25)
    quartiles_err_2_all=np.quantile(abs_err_all,0.5) 
    quartiles_err_3_all=np.quantile(abs_err_all,0.75)

    sign_err_all=y_modeled_zero-y_observed
    avg_sign_err_all=np.mean(sign_err_all)
    quartiles_sign_err_2_all=np.quantile(sign_err_all,0.5) 

    E90_all=np.quantile(abs_err_all,0.9)
    E95_all=np.quantile(abs_err_all,0.95)

    if predict_for_test: #variables in this area denoted by the prefix PT
        PT_x_norm=(x_test-min_input_m1)/(max_input_m1-min_input_m1)
        #predict based on m1
        PT_predicted_m1=model1_load.predict(PT_x_norm)
        #flag samples for m1 or intersection (later, we flag for m2)
        PT_y_norm_eval,PT_y_norm_eval_flag=clamp_vector(PT_predicted_m1,0,maxval_cut,np.zeros_like(PT_predicted_m1))

        #values were flagged,  
        #now let's select the registers for which the high concentration
        #model should be run
        [PT_size1, PT_size2]=PT_y_norm_eval_flag.shape

        PT_m1_x_eval=np.zeros([0,varnum])
        PT_m1_y_eval=np.zeros(0)
        PT_m1_data_coords=np.zeros([0,2])
        for i in range(0,PT_size1):
            if PT_y_norm_eval_flag[i,0]==0: #only m1
                temporary_x_eval=x_test[i,0:varnum]
                PT_m1_x_eval=np.row_stack((PT_m1_x_eval,temporary_x_eval))#
                PT_m1_y_eval = np.append(PT_m1_y_eval, y_test[i, 0])
        if PT_m1_x_eval.size > 0:
            PT_y_eval_m1=(PT_m1_y_eval-min_m1)/(max_m1-min_m1)
            PT_x_eval_m1=(PT_m1_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
            PT_predicted_m1=model1_load.predict(PT_x_eval_m1)
            PT_predicted_m1_scaled=PT_predicted_m1*(max_m1-min_m1)+min_m1
        else:
            PT_predicted_m1_scaled=[]   

        PT_mb_x_eval=np.zeros([0,varnum])
        PT_mb_y_eval=np.zeros(0)
        PT_mb_data_coords=np.zeros([0,2])
        PT_mi_x_eval=np.zeros([0,varnum])
        PT_mi_y_eval=np.zeros(0)
        PT_mi_data_coords=np.zeros([0,2])
        PT_m2_x_eval=np.zeros([0,varnum])
        PT_m2_y_eval=np.zeros(0)
        PT_m2_data_coords=np.zeros([0,2])
        PT_index_flagm2=np.empty((PT_size1))
        PT_index_flagm2[:] = np.nan
        PT_counter=0
        for i in range(0,PT_size1):
            if PT_y_norm_eval_flag[i,0]==1: #intersection, or m2 (needs to be defined again)
                temporary_x_eval=x_test[i,0:varnum]
                PT_mb_x_eval=np.row_stack((PT_mb_x_eval,temporary_x_eval))#
                PT_mb_y_eval = np.append(PT_mb_y_eval, y_test[i, 0])
                PT_index_flagm2[i]=PT_counter
                PT_counter=PT_counter+1
        if PT_mb_x_eval.size > 0:
            PT_y_eval_mb2=(PT_mb_y_eval-min_m2)/(max_m2-min_m2)
            PT_x_eval_mb2=(PT_mb_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
            PT_predicted_mb2=model2_load.predict(PT_x_eval_mb2)
            PT_predicted_mb2_scaled=PT_predicted_mb2*(max_m2-min_m2)+min_m2
            #flag samples for m2
            foo,PT_y_norm_eval_flag_m2=clamp_vector(PT_predicted_mb2,max_m1,max_m2,np.zeros_like(PT_predicted_mb2))
            for i in range(0,PT_size1):
                #important, indexes compatibility 
                PT_i_flagm2=PT_index_flagm2[i]
                if np.isnan(PT_i_flagm2)==False:
                    PT_i_flagm2_int=int(PT_i_flagm2)
                    if PT_y_norm_eval_flag_m2[PT_i_flagm2_int,0]==-1: #real intersection
                        temporary_x_eval=x_test[i,0:varnum]
                        PT_mi_x_eval=np.row_stack((PT_mi_x_eval,temporary_x_eval))#
                        PT_mi_y_eval = np.append(PT_mi_y_eval, y_test[i, 0])
            if PT_mi_x_eval.size > 0:
                    #run m1
                    PT_y_eval_mi1=(PT_mi_y_eval-min_m1)/(max_m1-min_m1)
                    PT_x_eval_mi1=(PT_mi_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
                    PT_predicted_mi1=model1_load.predict(PT_x_eval_mi1)
                    PT_predicted_mi1_scaled=PT_predicted_mi1*(max_m1-min_m1)+min_m1
                    #run m2
                    PT_y_eval_mi2=(PT_mi_y_eval-min_m2)/(max_m2-min_m2)
                    PT_x_eval_mi2=(PT_mi_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
                    PT_predicted_mi2=model2_load.predict(PT_x_eval_mi2)
                    PT_predicted_mi2_scaled=PT_predicted_mi2*(max_m2-min_m2)+min_m2
                    if predict_both_models:
                        #use both models
                        PT_predicted_mi_scaled=(PT_predicted_mi1_scaled+PT_predicted_mi2_scaled)/2.0 
                    else:
                        PT_predicted_mi_scaled=PT_predicted_mi2_scaled #use just m2
            else:
                    PT_predicted_mi_scaled=[] 
                
        for i in range(0,PT_size1):
            #important, indexes compatibility 
            PT_i_flagm2=PT_index_flagm2[i]
            if np.isnan(PT_i_flagm2)==False:
                PT_i_flagm2_int=int(PT_i_flagm2)
                if PT_y_norm_eval_flag_m2[PT_i_flagm2_int,0]>-1: #only m2
                    temporary_x_eval=x_test[i,0:varnum]
                    PT_m2_x_eval=np.row_stack((PT_m2_x_eval,temporary_x_eval))#
                    PT_m2_y_eval = np.append(PT_m2_y_eval, y_test[i, 0])
        if PT_m2_x_eval.size > 0:
            PT_y_eval_m2=(PT_m2_y_eval-min_m2)/(max_m2-min_m2)
            PT_x_eval_m2=(PT_m2_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
            PT_predicted_m2=model2_load.predict(PT_x_eval_m2)
            PT_predicted_m2_scaled=PT_predicted_m2*(max_m2-min_m2)+min_m2
        else:
            PT_predicted_m2_scaled=[] 

        PT_y_observed_step1=np.append(PT_m1_y_eval,PT_mi_y_eval)
        PT_y_observed=np.append(PT_y_observed_step1,PT_m2_y_eval)

        PT_y_modeled_step1=np.append(PT_predicted_m1_scaled,PT_predicted_mi_scaled)        
        PT_y_modeled=np.append(PT_y_modeled_step1,PT_predicted_m2_scaled)      
            
        #no concentration value below zero
        PT_y_modeled_zero = PT_y_modeled.copy()
        PT_y_modeled_zero[PT_y_modeled_zero < 0] = 0
        
        #get output back from log scale
        PT_y_modeled_zero_exp=np.exp(PT_y_modeled_zero)
        PT_y_observed_exp=np.exp(PT_y_observed)
        
        #statistics for all of the csv given. Attention, can mix train, val, and test
        PT_rmse_2models_all=np.sqrt(np.sum(np.absolute(PT_y_modeled_zero_exp.flatten()-PT_y_observed_exp.flatten()))**2.0)/len(PT_y_observed_exp.flatten())

        PT_mae_all=sum(abs(PT_y_modeled_zero_exp-PT_y_observed_exp))/len(PT_y_modeled_zero_exp)

        PT_max_err_all=max(abs(PT_y_modeled_zero_exp-PT_y_observed_exp))

        PT_abs_err_all=abs(PT_y_modeled_zero_exp-PT_y_observed_exp)
        
        #quartiles of error
        PT_quartiles_err_1_all=np.quantile(PT_abs_err_all,0.25)
        PT_quartiles_err_2_all=np.quantile(PT_abs_err_all,0.5) 
        PT_quartiles_err_3_all=np.quantile(PT_abs_err_all,0.75)

        PT_sign_err_all=PT_y_modeled_zero_exp-PT_y_observed_exp
        PT_avg_sign_err_all=np.mean(PT_sign_err_all)
        PT_quartiles_sign_err_2_all=np.quantile(PT_sign_err_all,0.5) 

        PT_E90_all=np.quantile(PT_abs_err_all,0.9)
        PT_E95_all=np.quantile(PT_abs_err_all,0.95)


    print('SSC Model Ran')
