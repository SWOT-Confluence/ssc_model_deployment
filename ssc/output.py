"""
Description

"""

# Standard imports
import os

# Third-party imports
import pandas as pd
import numpy as np

# Local imports

# Functions

def output():
    pass

def feature_output(feature_dict, out_dir, cloud_cover, mgrs_flag, date, l_or_s, args, lat, lon, filename):


    df = pd.DataFrame(data=np.array(list(feature_dict.values())).T, columns=list(feature_dict.keys()))
    print('ouputting results to csv...')
    print(df.head())
    df['cloud_cover'] = cloud_cover 
    df['date'] = date
    df['LorS'] = l_or_s
    df['MGRS'] = mgrs_flag
    df['lat'] = lat
    df['lon'] = lon

    # if args.training:
    #     old_training_data = pd.read_csv('/data/input/ssc/sun_test_set.csv')
    #     tss_values = []
    #     relative_days = []
    #     site_ids = []
    #     for i in lat:
    #         this_site = old_training_data[(old_training_data['lat']==i) & (old_training_data['date']==df['date'].iloc[0])]
    #         assert len(this_site) == 1
    #         tss_value = list(set(this_site['tss_value'].unique()))
    #         relative_day = list(set(this_site['relative_day'].unique()))
    #         site_id = list(set(this_site['SiteID'].unique()))
    #         tss_values.append(tss_value)
    #         relative_days.append(relative_day)
    #         site_ids.append(site_id)
    #     df['tss_value'] = tss_values
    #     df['relative_day'] = relative_days
    #     df['site_id'] = site_ids
        


    # df.to_csv(os.path.join(out_dir,filename.replace('.tar','') + '.csv'))
    return df