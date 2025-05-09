# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:52:46 2024

Test the function ann_ssc_model

@author: Luisa Lucchese
"""

import pandas as pd

from ann_ssc_model_v2 import ann_ssc_model

# #load models
m1_path='./ssc_models/m1'
m2_path='./ssc_models/m2'

predict_both_models_bool=True #True if you want to use both m1 and m2 in the range
    # that they're both valid.
    
ann_ssc_model_outpath='./'

preprocessed_dataframe=pd.read_csv("D:/Luisa/meetings_AIST/meeting_feb_2024/points_to_extract/allpoints_merged/SUN_CREATEDMISSI.csv")

status= ann_ssc_model(preprocessed_dataframe, ann_ssc_model_outpath, m1_path, m2_path, predict_both_models_bool)

print(status)

