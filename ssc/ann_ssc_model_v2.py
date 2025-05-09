# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:59:37 2023

@author: Luisa Lucchese

Execute a model composed of m1 and m2 in certain samples
and save the output.
"""

import numpy as np
import tensorflow as tf
import csv
import pickle


# #load models

# m1_path='./ssc_models/m1'
# m2_path='./ssc_models/m2'

# predict_both_models_bool=True #True if you want to use both m1 and m2 in the range
#     # that they're both valid.

def ann_ssc_model(features_for_ann_model:dict, path_load_m1 = '/data/input/ssc/ssc_models/m1', path_load_m2 = '/data/input/ssc/ssc_models/m1', predict_both_models= False):
   
    # path_load_m1=m1_path
    # path_load_m2=m2_path

    # Set true if you want to use both m1 and m2 in the range that they are both valid.
    # predict_both_models=predict_both_models_bool

    #the features used in the models. Alter accordingly.
    # input_cols= ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
    #                 'b2_median','b3_min','b3_mean','b3_max','b3_std','b3_median','b4_min','b4_mean','b4_max','b4_std',\
    #                 'b4_median','b10_min','b10_mean','b10_max','b10_std',\
    #                             'b10_median','b11_min','b11_mean','b11_max','b11_std','b11_median','b12_min','b12_mean','b12_max','b12_std',\
    #                                 'b12_median','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','LorS']

    input_cols = list(features_for_ann_model.keys())

    # Clear all previously registered custom objects and load loss function
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
                               
    model1_load = tf.keras.models.load_model(path_load_m1)
    model2_load = tf.keras.models.load_model(path_load_m2)
    
    #loading variables
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

    
    input_cols= ['b1_min','b1_mean','b1_max','b1_std','b1_median','b1_count','b2_min','b2_mean','b2_max','b2_std',\
                    'b2_median','b3_min','b3_mean','b3_max','b3_std','b3_median','b4_min','b4_mean','b4_max','b4_std',\
                    'b4_median','b10_min','b10_mean','b10_max','b10_std',\
                                'b10_median','b11_min','b11_mean','b11_max','b11_std','b11_median','b12_min','b12_mean','b12_max','b12_std',\
                                    'b12_median','b8a_min','b8a_mean','b8a_max','b8a_std','b8a_median','LorS']
    
    output_cols=['tss_value']
    
    coords_cols= ['lat','lon','date']
    
    #limits for m1
    minimum_out=0 #no concentration below 0 is possible
    #----------------ONLY FOR DEV
    varnum=len(input_cols)
    # varnum=42
        #----------------ONLY FOR DEV
    
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
    
    
    
    # data_input=preprocessed_dataframe[input_cols]
    print(np.array(list(features_for_ann_model.values())).shape, 'preshape')
    data_input_array=np.array(list(features_for_ann_model.values())).T
    print(data_input_array.shape, 'postshape')
    
    data_input_array_noinf=data_input_array
    #remove samples with infinite values 
    data_input_array_noinf[data_input_array_noinf == float('+inf')]=float('nan')
    
    #filter input
    locations_nan_input=np.isnan(data_input_array_noinf).any(axis=1)
    
    data_input_array_filtered=data_input_array[~locations_nan_input]
    
    # data_output=preprocessed_dataframe[output_cols]
    # data_output_array=data_output.to_numpy()
    
    data_output_array_filtered=data_output_array[~locations_nan_input]
    
    # #filter output
    locations_nan_output=np.isnan(data_output_array_filtered).any(axis=1)
    
    data_input_array_filtered2=data_input_array_filtered[~locations_nan_output]

    #-----------ONLY FOR DEVELOPMENT-----------------------
    # data_input_array_filtered2=data_input_array_filtered
    #----------------ONLY FOR DEVELOPMENT--------------

    # data_output_array_filtered2=data_output_array_filtered[~locations_nan_output]
    
    # data_coords=preprocessed_dataframe[coords_cols]
    # data_coords_array=data_coords.to_numpy()
    # data_coords_array_filtered=data_coords_array[~locations_nan_input]
    # data_coords_array_filtered2=data_coords_array_filtered[~locations_nan_output]
    
    #normalize    
    
    x_norm=(data_input_array_filtered2-min_input_m1)/(max_input_m1-min_input_m1)
    
    #predict based on m1
    predicted_m1=model1_load.predict(x_norm)
    
    #flag samples for m2
    y_norm_eval,y_norm_eval_flag=clamp_vector(predicted_m1,0,maxval_cut,np.zeros_like(predicted_m1))
    
    #values were flagged, let's select the registers for which the high concentration
    #model should be ran
    [size1, size2]=y_norm_eval_flag.shape
    
    m2_x_eval=np.zeros([0,varnum])
    m2_y_eval=np.zeros(0)
    m2_data_coords=np.zeros([0,3])
    m2_data_coords_kept=np.zeros([0,3])
    m2_x_eval_kept=np.zeros([0,varnum])
    m2_y_eval_kept=np.zeros(0)
    for i in range(0,size1):
        if y_norm_eval_flag[i,0]==1:
            temporary_x_eval=data_input_array_filtered2[i,0:varnum]
            m2_x_eval=np.row_stack((m2_x_eval,temporary_x_eval))#
            # m2_y_eval = np.append(m2_y_eval, data_output_array_filtered2[i, 0])
            #drag coordinate info
            # m2_data_coords=np.row_stack((m2_data_coords,data_coords_array_filtered2[i, :]))
        else:
            temporary_x_eval_kept=data_input_array_filtered2[i,0:varnum]
            m2_x_eval_kept=np.row_stack((m2_x_eval_kept,temporary_x_eval_kept))#
            # m2_y_eval_kept = np.append(m2_y_eval_kept, data_output_array_filtered2[i, 0])
            # m2_data_coords_kept=np.row_stack((m2_data_coords_kept,data_coords_array_filtered2[i, :]))
    
    
    if predict_both_models:
        print('use both models at the common area')
        size_predicted_m1_for_common=0
        predicted_m1_for_common=np.zeros([1,1])
        to_run_m1=np.zeros([1,varnum])
        if m2_x_eval.size > 0: # test if the registers that should run model2 are not empty
            #y_eval_m2=(m2_y_eval-min_m2)/(max_m2-min_m2)
            x_eval_m2=(m2_x_eval-min_input_m2)/(max_input_m2-min_input_m2)# this one should actually be based on m2_x_train 
            predicted_m2=model2_load.predict(x_eval_m2) 
            predicted_m2_scaled=predicted_m2*(max_m2-min_m2)+min_m2
            #test if they are between the common area of the models
            for i in range(0,len(predicted_m2)):
                if predicted_m2_scaled[i,:] < maximum_out:
                    size_predicted_m1_for_common=size_predicted_m1_for_common+1
            predicted_m1_for_common=np.zeros([size_predicted_m1_for_common,1])
            count_predicted_m1_for_common=0
            #print(str(predicted_m2_scaled))
            for i in range(0,len(predicted_m2)):
                if predicted_m2_scaled[i,:] < maximum_out:
                    #run the first model too
                    to_run_m1[:,:]=x_eval_m2[i,:]
                    predicted_m1_for_common[count_predicted_m1_for_common]=(model1_load.predict(to_run_m1))*(maximum_out-minimum_out)+minimum_out
                    predicted_m2_scaled[i]=(predicted_m2_scaled[i]+predicted_m1_for_common[count_predicted_m1_for_common])/2.0
                    print ('used both models for i='+str(i)+' final prediction:' +str(predicted_m2_scaled[i]))
                    count_predicted_m1_for_common=count_predicted_m1_for_common+1
            #print(str(predicted_m2_scaled))        
            #predicted_m2_scaled=predicted_m2*(max_m2-min_m2)+min_m2 #repeat because predicted_m2 changed - no need
        else:
            predicted_m2_scaled=[]
    elif not predict_both_models:
        print('do not use both models, only model 2, at the common area')
        if m2_x_eval.size > 0:
            #y_eval_m2=(m2_y_eval-min_m2)/(max_m2-min_m2)
            x_eval_m2=(m2_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
            predicted_m2=model2_load.predict(x_eval_m2)
        
            predicted_m2_scaled=predicted_m2*(max_m2-min_m2)+min_m2
        else:
            predicted_m2_scaled=[]
    else:
        print('wrong option, please choose if predict with both models on the common area')
    
    #y_eval_m1_kept=(m2_y_eval_kept-minimum_out)/(maximum_out-minimum_out)
    x_eval_m1_kept=(m2_x_eval_kept-min_input_m1)/(max_input_m1-min_input_m1)#
    predicted_test_m1_kept=model1_load.predict(x_eval_m1_kept)
    
    # coords_obs=np.row_stack((m2_data_coords,m2_data_coords_kept))
    
    predicted_m1_scaled=predicted_test_m1_kept*(maximum_out-minimum_out)+minimum_out
    
    y_modeled=np.exp(np.append(predicted_m2_scaled,predicted_m1_scaled))
    
    #no concentration value below zero
    y_modeled_zero = y_modeled.copy()
    y_modeled_zero[y_modeled_zero < 0] = 0
    
    #to make maps
    # y_mod_and_coord=np.column_stack((coords_obs,y_modeled_zero))
    
    # #save to csv file
    # makecsvname=(ann_ssc_model_outpath+'/'+'ssc_prediction'+'.csv')#
    # columns = ['lat', 'lon', 'date', 'SSC_modeled']
    # rownumber,foo=y_mod_and_coord.shape
    # with open(makecsvname, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(columns)
    #     for row in range(0,rownumber):
    #         row_to_write=y_mod_and_coord[row]
    #         writer.writerow(row_to_write) 


    return y_modeled_zero

