# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:26:15 2023

@author: Luisa Lucchese

Read the Fmask and return the mask for cloud, cloud shadow, and snow/ice.
Aerossol and cirrus bits are reserved, but not used.

inputs:
    - output_raster_intermed: the path where the projected fmask is
    
outputs:
    - mask_final: array with nan where the mask should mask out the pixel,
    and 1 where it should not
"""

def read_Fmask(output_raster_intermed, maskedout):
    # I will start by basing myself on this code:
    # https://lpdaac.usgs.gov/resources/e-learning/getting-started-cloud-native-hls-data-python/
        
    from osgeo import gdal
    import osr
    import numpy as np
    
    #open the fmask file and read the array
    step1 = gdal.Open(output_raster_intermed, gdal.GA_ReadOnly)
    
    #get geotransformation
    GT_input = step1.GetGeoTransform()
    
    step2 = step1.GetRasterBand(1)
    fmask_array = step2.ReadAsArray()
    size1,size2=fmask_array.shape
    
    bitword_order = (1, 1, 1, 1, 1, 1, 2)  # set the number of bits per bitword
    num_bitwords = len(bitword_order)      # Define the number of bitwords based on your input above
    total_bits = sum(bitword_order)        # Should be 8, 16, or 32 depending on datatype
    
    qVals = list(np.unique(fmask_array))  # Create a list of unique values that need to be converted to binary and decoded
    all_bits = list()
    goodQuality = []
    for v in qVals:
        all_bits = []
        bits = total_bits
        i = 0
    
        # Convert to binary based on the values and # of bits defined above:
        bit_val = format(v, 'b').zfill(bits)
        print('fmask', '\n' + str(v) + ' = ' + str(bit_val))
        all_bits.append(str(v) + ' = ' + str(bit_val))
    
        # Go through & split out the values for each bit word based on input above:
        for b in bitword_order:
            prev_bit = bits
            bits = bits - b
            i = i + 1
            if i == 1:
                bitword = bit_val[bits:]
                print(' Bit Word ' + str(i) + ': ' + str(bitword))
                all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword)) 
            elif i == num_bitwords:
                bitword = bit_val[:prev_bit]
                print(' Bit Word ' + str(i) + ': ' + str(bitword))
                all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
            else:
                bitword = bit_val[bits:prev_bit]
                print(' Bit Word ' + str(i) + ': ' + str(bitword))
                all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
    
        # 2, 4, 5, 6 are the bits used. 2,4,5 should = 0 if no clouds, cloud shadows were present, and pixel is not snow/ice. 6 should be =1 indicating the presence of water.
        if int(all_bits[2].split(': ')[-1]) + int(all_bits[4].split(': ')[-1]) + \
        int(all_bits[5].split(': ')[-1]) == 0 and int(all_bits[6].split(': ')[-1]) == 1:
            goodQuality.append(v)
    
    
    seemask=np.ones(shape=(size1,size2))
    mask_final = np.ma.MaskedArray(seemask, np.in1d(fmask_array, goodQuality, invert=True))  # Apply QA mask to the EVI data
    mask_final = np.ma.filled(mask_final, np.nan)                                                 # Set masked data to nan
    
    #save the mask just so we know what is being masked out 
    #I should comment the part below before heading to production
    driver = gdal.GetDriverByName( 'GTiff' )
    # maskedout = '/media/travis/work/repos/hls_full/aist-hls-data/outputs/maskedout.tif'
    dst_ds=driver.Create(maskedout,size2,size1,1,gdal.GDT_Float64)
    dst_ds.SetGeoTransform(GT_input)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'WGS84' )
    dst_ds.SetProjection( srs.ExportToWkt() )
    dst_ds.GetRasterBand(1).WriteArray( mask_final ) #.astype(np.float32)
    
    #step1=None
    
    return mask_final

