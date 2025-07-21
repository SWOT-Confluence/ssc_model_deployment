import pandas as pd
import glob
import os
import numpy as np
from netCDF4 import num2date
import datetime
import logging

def calculate_sedflux(model_outputs_df, consensus_dir:str):
    print('starting sedflux')
    logging.info('starting sedflux...')

    # init column
    model_outputs_df["sedflux"] = np.nan
    model_outputs_df["flpe_consensus"] = np.nan


    for a_reach_id in list(model_outputs_df['reach_id'].unique()):
        logging.info(f'sedflux on {a_reach_id}...')
        ssc_reach_data = model_outputs_df[model_outputs_df['reach_id'] == a_reach_id]
        try:
            logging.info('Trying sedflux...')
            
            # validation_reach_data = ncf.Dataset(os.path.join(validation_dir, f'{a_reach_id}_validation.nc'))
            consensus_reach_data = pd.read_csv(os.path.join(consensus_dir, f'{a_reach_id}_consensus.csv'))


            # Assume: time_var is the netCDF4 Variable 
            #         date_x is a list or array of date strings like "2023-05-26"

            consensus_date = consensus_reach_data['date']
            ssc_date = ssc_reach_data['date'].values


            # # Convert the time variable (ignoring -9999 values)
            # valid_time_mask = (time_var[:] != -9999)
            # valid_times = time_var[valid_time_mask]
            # time_dates = num2date(valid_times, units=time_var.units)

            # # Convert to pure date (no time part)
            # time_dates = np.array([d.date() for d in time_dates])

            # # Keep original indices for valid times
            # valid_indices = np.where(valid_time_mask)[0]

            # Convert date_x to datetime.date objects
            ssc_date_dt = [datetime.datetime.strptime(str(d), "%Y-%m-%d").date() for d in ssc_date]
            consensus_date_dt = [datetime.datetime.strptime(str(d), "%Y-%m-%d").date() for d in consensus_date]

            # Store matches
            matches = []

            for d in ssc_date_dt:
                # Create date range: ±1 day
                date_range = {d - datetime.timedelta(days=1), d, d + datetime.timedelta(days=1)}
                
                # Find all indices in time_dates where the date is in date_range
                match = [t for i, t in enumerate(consensus_date_dt) if t in date_range]
                matches.append(match)

            # matches will be a list of index lists for each date_x entry
            logging.info('MATCHES HERE...')
            logging.info(matches)

            # This will store the final sedflux values for each row in SSC data
            sedflux_values = []

            for i, match_dates in enumerate(matches):
                if match_dates:  # we have at least one ±1-day match
                    # You could take the average median across matches, or pick the first one
                    matched_medians = [consensus_reach_data.loc[j, "median"] for j, t in enumerate(consensus_date_dt) if t in match_dates]
                    
                    # Filter out NAs
                    matched_medians = [m for m in matched_medians if pd.notnull(m)]
                    
                    if matched_medians:
                        avg_median = np.mean(matched_medians)
                        sedflux = avg_median * ssc_reach_data.iloc[i]["SSC"]  * 0.001
                    else:
                        sedflux = np.nan
                else:
                    sedflux = np.nan

                sedflux_values.append(sedflux)

            # Add the new column to your dataframe
            # ssc_reach_data["sedflux"] = sedflux_values

            model_outputs_df.loc[ssc_reach_data.index, "sedflux"] = sedflux_values
            model_outputs_df.loc[ssc_reach_data.index, "flpe_consensus"] = avg_median


        except Exception as e:
            logging.info('No file')
            logging.info(e)
    return model_outputs_df
    



