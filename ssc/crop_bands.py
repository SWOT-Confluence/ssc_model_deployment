import os
import traceback

# 3rd party imports
from osgeo import gdal
import rasterio as rio
import utm
from shapely.geometry import Polygon
from shapely.geometry import mapping
from pyproj import CRS
import geopandas
import traceback
import pyproj
from PIL import Image


    
    # local imports
from ssc.extract_info_HLS_filename import extract_info_HLS_filename
from ssc.build_HLS_filenames import build_HLS_filenames
from ssc.build_HLS_filenames_extrabands import build_HLS_filenames_extrabands

import numpy as np

def crop_around_index(array, index, fill_value=-9999):
    half_size = 256  # Half of the desired crop size
    index_row, index_col = index
    # array = array[0]
    
    # Determine start and end indices for cropping
    start_row = int(max(0, index_row - half_size))
    end_row = int(min(array.shape[0], index_row + half_size))
    start_col = int(max(0, index_col - half_size))
    end_col = int(min(array.shape[1], index_col + half_size))
    
    # Create a new array filled with the fill value
    cropped_array = np.full((512, 512), fill_value) #nan
    
    # Calculate the region to copy from the original array
    source_start_row = int(half_size - min(index_row, half_size))
    source_end_row = int(half_size + min(array.shape[0] - index_row, half_size))
    source_start_col = int(half_size - min(index_col, half_size))
    source_end_col = int(half_size + min(array.shape[1] - index_col, half_size))

    try:
        # the raw index of the start and end of rows and cols in a cookie cutter kind of way
        cookie_start_top=int(index_row)-half_size
        cookie_end_bottom=int(index_row)+half_size
        cookie_start_left=int(index_col)-half_size
        cookie_end_right=int(index_col)+half_size

        # Copy the relevant portion from the original array to the cropped array
        first_array = array[start_row:end_row, start_col:end_col]
        #second_array = cropped_array[source_start_row:source_end_row, source_start_col:source_end_col] #nan
        # add padding 
        left_pad=0
        right_pad=0
        top_pad=0
        bottom_pad=0
        if cookie_start_top < 0:
            # print('cookie_start_top<0, cookie_start_top='+str(cookie_start_top))
            top_pad= 0-cookie_start_top
        if cookie_end_bottom > array.shape[0]:
            # print('cookie_end_bottom > array.shape[0], cookie_end_bottom='+str(cookie_end_bottom)+' array.shape[0]='+str(array.shape[0]))
            bottom_pad=cookie_end_bottom-array.shape[0]
        if cookie_start_left<0:
            left_pad= cookie_start_left
        if cookie_end_right > array.shape[1]:
            right_pad=cookie_end_right-  array.shape[1]
        # Apply padding
        padded_array = np.pad(first_array, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=fill_value)
        if left_pad>0 or right_pad>0 or top_pad>0 or bottom_pad>0:
            print('top_pad is '+str(top_pad)+', bottom_pad is '+str(bottom_pad)+', left_pad is '+str(left_pad)+', right_pad is '+str(right_pad))
        
#[source_start_row:start_row]
    
        #cropped_array[source_start_row:source_end_row, source_start_col:source_end_col] = padded_array #first_array
        cropped_array[:, :] = padded_array
    except Exception as e:
        cropped_array[source_start_row:source_end_row, source_start_col:source_end_col] = -9999
        print('Band crop failed, using fill value...')
        print(e)
    
    return cropped_array

# Example usage:
# array = your_array_here
# index = (row_index, col_index)
# cropped_array = crop_around_index(array, index)


def crop_bands(all_bands_in_memory, node_data, filename, buffersize, l_or_s):

    
    # filename = 'D:/Luisa/start_insitu_data_wrongdata/downloaded/HLS.S30.T18TWK.2021250T154809.v2.0.B01.subset.tif'
    # pointlon=-73.9978
    # pointlat=40.23028
    
    # siteid='MNPCA-69-0249-00-102' #the unique index of the site, needed to save the files correctly.
    # print('all bands before cropping')

    pointlat = node_data[2][1]
    pointlon = node_data[2][0]
    
    # print(pointlat, pointlon)
    siteid = '_'.join([str(node_data[1]), str(node_data[0])])
    pathtocropped='.'

    nbands = len(all_bands_in_memory)
    
    # buffersize=500 #500 m
    
    actual_filename, path_to_file, tile, date, time, source=extract_info_HLS_filename(filename)
    # print(tile, date, time)
    
    #now, reproject this into the same grid of the HLS.


    #get location and size of the file
    GT_input = all_bands_in_memory[0].rio.transform()
    # print(GT_input)
    
    #produce area around in situ to be saved
    # print('this is what is going in', float(pointlat), float(pointlon))
    (eastingpt, northingpt, zone_number, zone_letter)=utm.from_latlon(float(pointlat), float(pointlon))
    
    if zone_letter in 'NPQRSTUVWX':
        zone_desig = 'north'
    else:
        zone_desig = 'south'
    
    crs = CRS.from_string(f'+proj=utm +zone={zone_number} +{zone_desig}')
    wsg = crs.to_authority()
    # print('northing and easting before reproj', eastingpt, northingpt)
    transformer = pyproj.Transformer.from_crs(wsg, all_bands_in_memory[0].rio.crs, always_xy=True)
    eastingpt, northingpt = transformer.transform(eastingpt, northingpt)
    # print('norhting and easting after reproj', eastingpt, northingpt)

    # print('zone sample zone number and letter', zone_number, zone_letter)
    
    # print('tile projection', all_bands_in_memory[0].rio.crs)

    
    minx_is=eastingpt-buffersize
    miny_is=northingpt-buffersize
    maxx_is=eastingpt+buffersize
    maxy_is=northingpt+buffersize
    
    #initializing the new_bounds array
    new_bounds = [
        (minx_is, miny_is),
        (maxy_is, maxx_is),
        (maxx_is, miny_is),
        (miny_is, minx_is)
    ]   
    # print('cropping images, new bounds: '+str(new_bounds))
    # dest, src, options

    

    
    geometry = Polygon(new_bounds) #not used
    # print('crs to authority from the sample point', wsg)
    # print('reprojecting point to tile')
    
    
    
    # project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    # utm_point = transform(project, geometry)

    # d = {'geometry': [geometry]}
    # gdf = geopandas.GeoDataFrame(d)
    # gdf = gdf.set_crs(wsg)
    # feature = gdf.iloc[0]
    # feature = [mapping(geometry)]
    # print(feature, 'before')

    # feature = [geometry]
    # print(feature, 'after')
    width = 512

    cropped_bands_in_memory = []
    x, y = (eastingpt, northingpt)
    # row, col = all_bands_in_memory[0].rio.index(x, y)
    col, row = ~all_bands_in_memory[0].rio.transform() * (x,y)
    # col = int(col)
    # row = int(row)

    # print('this is the row and column we are trying to crop around...', row,col)
    # given pixel coords get geo cords
    # print('here is the coordinate at the center of the whole image...', all_bands_in_memory[0].xy(all_bands_in_memory[0].height // 2, all_bands_in_memory[0].width // 2))
    nx = all_bands_in_memory[0].shape[1]
    ny = all_bands_in_memory[0].shape[0]
    # print(all_bands_in_memory[0])
    # print('here are image bounds', nx, ny)
    
    idx_left  = max(col - width, 0)
    idx_right = min(col + width + 1, nx - 1)
    if idx_right == nx - 1:  # When the last index in x-dimension is selected
        # print('out of x dim')
        idx_right = None
    # else:
    #     print('in x dim')
        
    idx_bot   = max(row - width, 0) 
    idx_top   = min(row + width + 1, ny - 1)
    if idx_top == ny - 1:    # When the last index in y-dimension is selected
        # print('out of y dem')
        idx_top = None
    # else:
    #     print('in y dem')
    # Output
    
        
    #run band loop
    #regular band loop
    for iband in range(nbands):
        try:
            # filename_extract=build_HLS_filenames(date,time,pointlon,pointlat,path_to_file,iband, source)
            # #using GDAL Translate
            # # http://gdal.org/python/osgeo.gdal-module.html#Translate
            # output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
            # cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)

            # cropHLS, out_transform = rio.mask.mask(all_bands_in_memory[iband], feature, crop=True)
            a_band = all_bands_in_memory[iband].values
            # print('here is a band that will become a view ', a_band)
            # print('here is the data at sample point...', a_band[row, col])
            
            # view = a_band[idx_bot:idx_top, idx_left:idx_right]
            try:
                view = crop_around_index(a_band, (row, col), fill_value=-9999)
            except Exception as e:

                print('Cropping failed...')

                print(e)
                # exit()
                raise ValueError(print('Cropping failed...'))
            # print('here is a view', view)
            cropped_bands_in_memory.append(view)


            
            
            
            
            
            
            # --------------------------------------------
            # mask, transform, cropHLS = rio.mask.raster_geometry_mask(all_bands_in_memory[iband], feature, crop=True)
            # try:
            #     im = Image.fromarray(all_bands_in_memory[iband].read(1))
            #     im.save("/data/input/ssc/before_cropping.png")
            # except:
            #     pass
            # cropHLS = cropHLS.crop(512,512)
            # cropped_bands_in_memory.append(all_bands_in_memory[iband].read(1, window = cropHLS))
            # print(cropped_bands_in_memory[0][0], 'here is the cropped band')
            # ---------------------------------------
            
            
            
            # im = Image.fromarray(cropped_bands_in_memory[0])
            # im.save("/data/input/ssc/after_cropping.png")
            # exit()

            # print(all_bands_in_memory[iband].read(1, window = cropHLS))
            # print(mask)

        except Exception as e:
        #    print('Could crop HLS, band number: ', str(iband))
           print(e)
           traceback.print_exc()
        #    exit()
           raise    ValueError('Could crop HLS, band number: ', str(iband))       
    
    # #8A Band
    # iband=iband+1
    # try:
    #     filename_extract=build_HLS_filenames_extrabands(date,time,pointlon,pointlat,path_to_file,source,'B8A')
    #     output_crop=os.path.join(pathtocropped,'croppedHLS_'+source+'_'+siteid+'_'+date+'_'+str(iband).zfill(2)+'.tif')
    #     cropHLS=gdal.Translate(output_crop, filename_extract, projWin = new_bounds)
    #     cropHLS=None
    # except Exception as e:
    #    print(f'Could not read HLS, band number: 8A')
    #    print(e)



    return cropped_bands_in_memory