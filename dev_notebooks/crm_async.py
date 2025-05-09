from multiprocessing import Pool
from pystac_client import Client  
from shapely.geometry import Polygon, LineString, MultiPoint, Point



data_dir = '/home/travis/data/10-24/mnt/input/sword'
reach_id = 23216000521


import glob
import netCDF4
import os
import numpy as np

data_dir = '/home/confluence/data/mnt/input/sword'


def get_reach_node_cords(data_dir, reach_id):

    all_nodes = []

    files = glob.glob(os.path.join(data_dir, '*v15.nc'))
    print(f'Searching across {len(files)} continents for nodes...')

    for i in files:

        rootgrp = netCDF4.Dataset(i, "r", format="NETCDF4")

        node_ids_indexes = np.where(rootgrp.groups['nodes'].variables['reach_id'][:].data.astype('U') == str(reach_id))

        if len(node_ids_indexes[0])!=0:
            for y in node_ids_indexes[0]:

                lat = str(rootgrp.groups['nodes'].variables['x'][y].data.astype('U'))
                lon = str(rootgrp.groups['nodes'].variables['y'][y].data.astype('U'))
                all_nodes.append([lat,lon])



            # all_nodes.extend(node_ids[0].tolist())

        rootgrp.close()

    print(f'Found {len(all_nodes)} nodes...')
    return all_nodes



# all_links =  find_download_links_for_reach_tiles(data_dir, reach_id)
all_points = get_reach_node_cords(data_dir, reach_id)

print(all_points)




# def  f(x):
#     collections = ['HLSL30.v2.0', 'HLSS30.v2.0']

#     STAC_URL = 'https://cmr.earthdata.nasa.gov/stac'

#     catalog = Client.open(f'{STAC_URL}/LPCLOUD/')

#     point = Point('0.22797328583233897','45.02741982760003')
#     search = catalog.search(collections=collections, intersects = point, datetime='2018-01-01T00:00:00Z/2018-02-01T23:59:59Z')
#     # search = catalog.search(collections=collections, datetime='2018-01-02T00:00:00Z/2018-01-02T23:59:59Z')   # date_range = '2018-01-01T00:00:00Z/2018-02-01T23:59:59Z')

#     item_collection = search.item_collection()

#     links =[]
#     for i in item_collection:
#         # print(i.assets)
#         for key in i.assets:
#             if key.startswith('B'):
#                 # link = i.assets[key].href.replace('https://data.lpdaac.earthdatacloud.nasa.gov/', 's3://')
#                 link = i.assets[key].href

#                 # print(link)
#                 links.append(link)



# pool = Pool(processes=4)              # start 4 worker processes
# result = pool.apply_async(f, [10])    # evaluate "f(10)" asynchronously
#         # prints "100" unless your computer is *very* slow
# pool.map(f, range(10))

# pool.close()