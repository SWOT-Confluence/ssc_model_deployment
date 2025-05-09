# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:01:51 2023

Clamp functions

@author: Luisa Lucchese
"""
#from auxfunctions.f_clamp import clamp_vector

import numpy as np

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

#cap values
def clamp_up(n,minn, maxn,flag,maxval_cut):
    shp1,shp2=np.shape(n)
    flag=np.zeros_like(n)
    for i in range(0,shp1):
        for j in range(0,shp2):
            if n[i,j]  > maxn*maxval_cut:
                n[i,j]= maxn
                flag[i,j]=1
            if n[i,j] < minn:
                n[i,j]= minn
                flag[i,j]=-1
            elif n[i,j]  > maxn:
                n[i,j]= maxn
                flag[i,j]=2            
    return n,flag

def clamp(n,minn, maxn,flag):
    shp1,shp2=np.shape(n)
    flag=np.zeros_like(n)
    for i in range(0,shp1):
        for j in range(0,shp2):
            if n[i,j] < minn:
                n[i,j]= minn
                flag[i,j]=-1
            elif n[i,j]  > maxn:
                n[i,j]= maxn
                flag[i,j]=1            
    return n,flag

def nan_max(n):
    shp1,shp2=np.shape(n)
    for i in range(0,shp1):
        for j in range(0,shp2):
            if np.isnan(n[i,j]) :
                n[i]=1
    return n