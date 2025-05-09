# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:54:58 2023

Separate function, executes ensemble model and returns values

@author: Luisa Lucchese
"""
import numpy as np

#from auxfunctions.f_execens import f_execens
from .f_clamp import clamp_vector

def f_execens_separated(model,model2,x_test,y_test,coords_test,minimum_out,maximum_out,min_m2,max_m2,min_input_m1,max_input_m1,min_input_m2,max_input_m2,maxval_cut,varnum):
    #start by running m1
    #
    #y_test_eval_m1=(y_test-minimum_out)/(maximum_out-minimum_out)
    x_test_eval_m1=(x_test-min_input_m1)/(max_input_m1-min_input_m1)#linear_scaling(x_test,min_input,max_input)#(x_test-min_input)/(max_input-min_input)
    
    predicted_test_m1=model.predict(x_test_eval_m1)
    
    y_test_norm_eval,flag_y_test_norm_eval=clamp_vector(predicted_test_m1,0,maxval_cut,np.zeros_like(predicted_test_m1))
  
    #values were flagged, let's select the registers for which the high concentration
    #model should be ran
    [size1, size2]=flag_y_test_norm_eval.shape
    
    m2_x_test_eval=np.zeros([0,varnum])
    m2_y_test_eval=np.zeros(0)
    m2_x_test_eval_kept=np.zeros([0,varnum])
    m2_y_test_eval_kept=np.zeros(0)
    m2_coords_test=np.zeros([0,2])
    m2_coords_test_kept=np.zeros([0,2])
    for i in range(0,size1):
        if flag_y_test_norm_eval[i,0]==1:
            temporary_x_test_eval=x_test[i,0:varnum]
            m2_x_test_eval=np.row_stack((m2_x_test_eval,temporary_x_test_eval))#np.append(m2_x_train,temporary_x_train,axis=0)
            m2_y_test_eval = np.append(m2_y_test_eval, y_test[i, 0])
            m2_coords_test = np.append(m2_coords_test, coords_test[i, :])
        else:
            temporary_x_test_eval_kept=x_test[i,0:varnum]
            m2_x_test_eval_kept=np.row_stack((m2_x_test_eval_kept,temporary_x_test_eval_kept))#
            m2_y_test_eval_kept = np.append(m2_y_test_eval_kept, y_test[i, 0])
            m2_coords_test_kept = np.append(m2_coords_test_kept, coords_test[i, :])
            
    # test if the registers that should run model2 are not empty
    if m2_x_test_eval.size > 0:
        #y_test_eval_m2=linear_scaling(m2_y_test_eval,min_m2,max_m2)#(m2_y_test_eval-min_m2)/(max_m2-min_m2)
        x_test_eval_m2=(m2_x_test_eval-min_input_m2)/(max_input_m2-min_input_m2)#linear_scaling(m2_x_test_eval,min_input,max_input)#(m2_x_test_eval-min_input)/(max_input-min_input)
        predicted_test_m2=model2.predict(x_test_eval_m2)
    
        predicted_m2_scaled=predicted_test_m2*(max_m2-min_m2)+min_m2
    else:
        predicted_m2_scaled=[]
    
    #y_test_eval_m1_kept=(m2_y_test_eval_kept-minimum_out)/(maximum_out-minimum_out)
    x_test_eval_m1_kept=(m2_x_test_eval_kept-min_input_m1)/(max_input_m1-min_input_m1)#linear_scaling(m2_x_test_eval_kept,min_input,max_input)#m2_x_test_eval_kept-min_input)/(max_input-min_input)
    predicted_test_m1_kept=model.predict(x_test_eval_m1_kept)
        
    y_observed=np.exp(np.append(m2_y_test_eval,m2_y_test_eval_kept))
    
    predicted_m1_scaled=predicted_test_m1_kept*(maximum_out-minimum_out)+minimum_out
    
    y_modeled=np.append(predicted_m2_scaled,predicted_m1_scaled)
    
    coords_ordered=np.append(m2_coords_test,m2_coords_test_kept)
    
    #undo the log transformation
    y_modeled=np.exp(y_modeled)
    return y_modeled,y_observed,coords_ordered


def f_execens(model,model2,x_test,y_test,coords_test,minimum_out,maximum_out,min_m2,max_m2,min_input_m1,max_input_m1,min_input_m2,max_input_m2,maxval_cut,varnum):
   
    
    max_m1=maximum_out
    min_m1=minimum_out
    predict_both_models=True
    
    # force type
    x_test = x_test.astype(float)
    
    x_test_eval_m1=(x_test-min_input_m1)/(max_input_m1-min_input_m1)
    #predict based on m1
    predicted_m1=model.predict(x_test_eval_m1)

    #flag samples for m1 or intersection (later, we flag for m2)
    y_test_norm_eval,flag_y_test_norm_eval=clamp_vector(predicted_m1,0,maxval_cut,np.zeros_like(predicted_m1))

    #values were flagged,  
    #now let's select the registers for which the high concentration
    #model should be run
    [size1, size2]=flag_y_test_norm_eval.shape

    m1_x_eval=np.zeros([0,varnum])
    m1_y_eval=np.zeros(0)
    m1_data_coords=np.zeros([0,2])
    for i in range(0,size1):
        if flag_y_test_norm_eval[i,0]==0: #only m1
            temporary_x_eval=x_test[i,0:varnum]
            m1_x_eval=np.row_stack((m1_x_eval,temporary_x_eval))#
            m1_y_eval = np.append(m1_y_eval, y_test[i, 0])
            #drag coordinate info
            m1_data_coords=np.row_stack((m1_data_coords,coords_test[i, :]))
    if m1_x_eval.size > 0:
       y_eval_m1=(m1_y_eval-min_m1)/(max_m1-min_m1)
       x_eval_m1=(m1_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
       predicted_m1=model.predict(x_eval_m1)
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
        if flag_y_test_norm_eval[i,0]==1: #intersection, or m2 (needs to be defined again)
            temporary_x_eval=x_test[i,0:varnum]
            mb_x_eval=np.row_stack((mb_x_eval,temporary_x_eval))#
            mb_y_eval = np.append(mb_y_eval, y_test[i, 0])
            #drag coordinate info
            mb_data_coords=np.row_stack((mb_data_coords,coords_test[i, :]))
            #we need to also save the index for compatibility
            index_flagm2[i]=counter
            counter=counter+1
    if mb_x_eval.size > 0:
       y_eval_mb2=(mb_y_eval-min_m2)/(max_m2-min_m2)
       x_eval_mb2=(mb_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
       predicted_mb2=model2.predict(x_eval_mb2)
       predicted_mb2_scaled=predicted_mb2*(max_m2-min_m2)+min_m2
       #flag samples for m2
       foo,y_norm_eval_flag_m2=clamp_vector(predicted_mb2,max_m1,max_m2,np.zeros_like(predicted_mb2))
       for i in range(0,size1):
           #important, indexes compatibility 
           i_flagm2=index_flagm2[i]
           if np.isnan(i_flagm2)==False:
               i_flagm2_int=int(i_flagm2)
               if y_norm_eval_flag_m2[i_flagm2_int,0]==-1: #real intersection
                   temporary_x_eval=x_test[i,0:varnum]
                   mi_x_eval=np.row_stack((mi_x_eval,temporary_x_eval))#
                   mi_y_eval = np.append(mi_y_eval, y_test[i, 0])
                   #drag coordinate info
                   mi_data_coords=np.row_stack((mi_data_coords,coords_test[i, :]))
       if mi_x_eval.size > 0:
            #run m1
            y_eval_mi1=(mi_y_eval-min_m1)/(max_m1-min_m1)
            x_eval_mi1=(mi_x_eval-min_input_m1)/(max_input_m1-min_input_m1)#
            predicted_mi1=model.predict(x_eval_mi1)
            predicted_mi1_scaled=predicted_mi1*(max_m1-min_m1)+min_m1
            #run m2
            y_eval_mi2=(mi_y_eval-min_m2)/(max_m2-min_m2)
            x_eval_mi2=(mi_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
            predicted_mi2=model2.predict(x_eval_mi2)
            predicted_mi2_scaled=predicted_mi2*(max_m2-min_m2)+min_m2
            if predict_both_models:
                #use both models
                predicted_mi_scaled=0.6*predicted_mi1_scaled+0.4*predicted_mi2_scaled 
            else:
                predicted_mi_scaled=predicted_mi2_scaled #use just m2
       else:
            predicted_mi_scaled=[] 
    else:
       predicted_mi_scaled=[] 
               
    for i in range(0,size1):
        #important, indexes compatibility 
        i_flagm2=index_flagm2[i]
        if np.isnan(i_flagm2)==False:
            i_flagm2_int=int(i_flagm2)
            if y_norm_eval_flag_m2[i_flagm2_int,0]>-1: #only m2
                temporary_x_eval=x_test[i,0:varnum]
                m2_x_eval=np.row_stack((m2_x_eval,temporary_x_eval))#
                m2_y_eval = np.append(m2_y_eval, y_test[i, 0])
                #drag coordinate info
                m2_data_coords=np.row_stack((m2_data_coords,coords_test[i, :]))
    if m2_x_eval.size > 0:
       y_eval_m2=(m2_y_eval-min_m2)/(max_m2-min_m2)
       x_eval_m2=(m2_x_eval-min_input_m2)/(max_input_m2-min_input_m2)#
       predicted_m2=model2.predict(x_eval_m2)
       predicted_m2_scaled=predicted_m2*(max_m2-min_m2)+min_m2
    else:
       predicted_m2_scaled=[] 

    y_observed_step1=np.append(m1_y_eval,mi_y_eval)
    y_observed_log=np.append(y_observed_step1,m2_y_eval)
    y_observed=np.exp(y_observed_log)

    coords_obs_step1=np.row_stack((m1_data_coords,mi_data_coords))
    coords_ordered=np.row_stack((coords_obs_step1,m2_data_coords))

    y_modeled_step1=np.append(predicted_m1_scaled,predicted_mi_scaled)        
    y_modeled_log=np.append(y_modeled_step1,predicted_m2_scaled)      

    #bringing y_modeled back from log scale
    y_modeled=np.exp(y_modeled_log)
   
    
   
# =============================================================================
#     #start by running m1
#     #
#     #y_test_eval_m1=(y_test-minimum_out)/(maximum_out-minimum_out)
#     #x_test_eval_m1=(x_test-min_input_m1)/(max_input_m1-min_input_m1)#
#     
#     #predicted_test_m1=model.predict(x_test_eval_m1)
#     
#    # y_test_norm_eval,flag_y_test_norm_eval=clamp_vector(predicted_test_m1,0,maxval_cut,np.zeros_like(predicted_test_m1))
#   
#     #values were flagged, let's select the registers for which the high concentration
#     #model should be ran
#     #[size1, size2]=flag_y_test_norm_eval.shape
#     
#     m2_x_test_eval=np.zeros([0,varnum])
#     m2_y_test_eval=np.zeros(0)
#     m2_x_test_eval_kept=np.zeros([0,varnum])
#     m2_y_test_eval_kept=np.zeros(0)
#     m2_coords_test=np.zeros([0,2])
#     m2_coords_test_kept=np.zeros([0,2])
#     for i in range(0,size1):
#         if flag_y_test_norm_eval[i,0]==1:
#             temporary_x_test_eval=x_test[i,0:varnum]
#             m2_x_test_eval=np.row_stack((m2_x_test_eval,temporary_x_test_eval))#np.append(m2_x_train,temporary_x_train,axis=0)
#             m2_y_test_eval = np.append(m2_y_test_eval, y_test[i, 0])
#             m2_coords_test = np.append(m2_coords_test, coords_test[i, :])
#         else:
#             temporary_x_test_eval_kept=x_test[i,0:varnum]
#             m2_x_test_eval_kept=np.row_stack((m2_x_test_eval_kept,temporary_x_test_eval_kept))#
#             m2_y_test_eval_kept = np.append(m2_y_test_eval_kept, y_test[i, 0])
#             m2_coords_test_kept = np.append(m2_coords_test_kept, coords_test[i, :])
#             
#     # test if the registers that should run model2 are not empty
#     if m2_x_test_eval.size > 0:
#         #y_test_eval_m2=linear_scaling(m2_y_test_eval,min_m2,max_m2)#(m2_y_test_eval-min_m2)/(max_m2-min_m2)
#         x_test_eval_m2=(m2_x_test_eval-min_input_m2)/(max_input_m2-min_input_m2)#linear_scaling(m2_x_test_eval,min_input,max_input)#(m2_x_test_eval-min_input)/(max_input-min_input)
#         predicted_test_m2=model2.predict(x_test_eval_m2)
#     
#         predicted_m2_scaled=predicted_test_m2*(max_m2-min_m2)+min_m2
#     else:
#         predicted_m2_scaled=[]
#     
#     #y_test_eval_m1_kept=(m2_y_test_eval_kept-minimum_out)/(maximum_out-minimum_out)
#     x_test_eval_m1_kept=(m2_x_test_eval_kept-min_input_m1)/(max_input_m1-min_input_m1)#linear_scaling(m2_x_test_eval_kept,min_input,max_input)#m2_x_test_eval_kept-min_input)/(max_input-min_input)
#     predicted_test_m1_kept=model.predict(x_test_eval_m1_kept)
#         
#     y_observed=np.exp(np.append(m2_y_test_eval,m2_y_test_eval_kept))
#     
#     predicted_m1_scaled=predicted_test_m1_kept*(maximum_out-minimum_out)+minimum_out
#     
#     y_modeled=np.append(predicted_m2_scaled,predicted_m1_scaled)
#     
#     coords_ordered=np.append(m2_coords_test,m2_coords_test_kept)
#     
#     #undo the log transformation
#     y_modeled=np.exp(y_modeled)
# =============================================================================
    return y_modeled,y_observed,coords_ordered
