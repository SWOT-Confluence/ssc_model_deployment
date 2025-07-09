import pandas
import glob
import netCDF4 as ncf
import os
import numpy as np
from netCDF4 import num2date
import datetime

def calculate_sedflux(model_outputs_df, validation_dir:str):
    for a_reach_id in list(model_outputs_df['reach_id'].unique()):
        ssc_reach_data = model_outputs_df[model_outputs_df['reach_id'] == a_reach_id]
        try:
            validation_reach_data = ncf.Dataset(os.path.join(validation_dir, f'{a_reach_id}_validation.nc'))


            # Assume: time_var is the netCDF4 Variable 
            #         date_x is a list or array of date strings like "2023-05-26"

            time_var = validation_reach_data['time']
            date_x = ssc_reach_data['date_x'].values()


            # Convert the time variable (ignoring -9999 values)
            valid_time_mask = (time_var[:] != -9999)
            valid_times = time_var[valid_time_mask]
            time_dates = num2date(valid_times, units=time_var.units)

            # Convert to pure date (no time part)
            time_dates = np.array([d.date() for d in time_dates])

            # Keep original indices for valid times
            valid_indices = np.where(valid_time_mask)[0]

            # Convert date_x to datetime.date objects
            date_x_dt = [datetime.datetime.strptime(str(d), "%Y-%m-%d").date() for d in date_x]

            # Store matches
            matches = []

            for d in date_x_dt:
                # Create date range: ±1 day
                date_range = {d - datetime.timedelta(days=1), d, d + datetime.timedelta(days=1)}
                
                # Find all indices in time_dates where the date is in date_range
                match = [valid_indices[i] for i, t in enumerate(time_dates) if t in date_range]
                matches.append(match)

            # matches will be a list of index lists for each date_x entry
            print('MATCHES HERE--------------------')
            print(matches)
        except Exception as e:
            print('No file')
            print(e)
    return model_outputs_df
    



