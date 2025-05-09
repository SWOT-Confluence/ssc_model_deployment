# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:05:57 2023

@author: Luisa Lucchese

Extract information from filenames of HLS

input:
    - filename: whole name of the HLS file with path
    
outputs:
    - actual_filename: just the filename, no path
    - path_all: all the path to the filename
    - int(band_string): band number as an integer
    - tile: name of the tile of HLS
    - date: the date it refers to
    - time: time it refers to
    - version: always v2, but in case it ever changes, saves version too
"""


def extract_info_HLS_filename(filename):
    import copy
    import os
    

    #separate what is the path and what is the filename
    # path_file=filename.split('/')
    # actual_filename=path_file[-1]
    actual_filename = os.path.basename(filename)


    # print(actual_filename)
    # copy_path_file = copy.deepcopy(path_file)
    # path_all='/'.join(copy_path_file[0:-1])
    # print(path_all)
    path_all = os.path.dirname(filename)
    
    # #verification -- is this actually an HLS file?
    
    # nameprod=actual_filename[0:3]
    filename_list = actual_filename.split('.')
    nameprod = filename_list[0]
    print(filename_list)
    
    if (nameprod=='HLS'):
        
        source = filename_list[1][0]
    
    #     tile=actual_filename[9:14]
        tile = filename_list[2]
    #     date=actual_filename[15:22]
        date = filename_list[3].split('T')[0]
    #     time=actual_filename[23:29]
        time = filename_list[3].split('T')[1]
    #     version=actual_filename[30:34]
    #     version = ('.').join([filename_list[4], filename_list[5]])
    # #     band_string=actual_filename[36:38]
    #     band_string = filename_list[6][1:].split('_')[0]

        
    # else:
    #     print('NOT HLS -- something went wrong with file ' + filename)
        
    return actual_filename, path_all, tile, date, time, source