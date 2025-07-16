# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:06:51 2025

@author: Luisa Lucchese
"""
import numpy
mid_x, mid_y = 512//2, 512//2
radius_mask = 10 # 10 pixels - only in UTM projected dataset
mask = numpy.empty((20,20))
mask[:] = numpy.nan
number_points=0
for i in range(0,radius_mask*2+1):
    for j in range(0,radius_mask*2+1):
        # print(i)
        xmap=i-10+0.5
        ymap=j-10+0.5
        radius_inner=xmap*xmap+ymap*ymap
        # print(radius_inner)
        if radius_inner < radius_mask*radius_mask:
            mask[i,j]=1.0 #numpy.nan
            number_points=number_points+1
            
            