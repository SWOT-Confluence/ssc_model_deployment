"""
Execute end to end processing of HLS tiles for SSC prediction.

Command line arguments:
    -i: index to locate tile in JSON file
    -j: Path to JSON file with list S3 paths to HLS tiles
    -o: output directory
    -r: Filepath to json containing reaches of interest

Example Deployment: 

sudo docker run -v /media/travis/work/data/mnt/input:/data/input ssc -i 0 -j hls_links_na.json -o . -r reaches.json -d /data/input

sudo docker run -v /media/travis/work/data/mnt/input:/data/input -v /home/travis:/home ssc -i 0 -j hls_links_na.json -o . -r reaches.json -d /data/input
"""
# python3 process_ssc.py -i 0 -j hls_links_na.json -o . -r reaches.json -d /media/travis/work/data/mnt/input

# sudo docker run -v /media/travis/work/data/mnt/input:/data/input -v /home/travis:/root/ ssc -i 0 -j hls_links_na.json -o /mnt/external/data/input/ssc -r reaches.json -d /data/input

# sudo docker run -v /mnt/external/data/input:/data/input -v /home/travis:/root/ ssc -i 0 -j hls_links_na.json -o /mnt/external/data/input/ssc -r reaches.json -d /data/input

# sudo docker run -v /mnt/input/:/data/input -v /home/ec2-user/:/root/ ssc -i 0 -j /data/input/ssc/training_data_tiles.json -o /mnt/external/data/input/ssc -r reaches.json -d /data/input --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar --latlon /data/input/ssc/training_data_lon_lat.json

# sudo docker run -v /mnt/input/:/data/input -v /home/ec2-user/:/root/ ssc -i 0 -j /data/input/ssc/training_data_tiles.json -o /mnt/external/data/input/ssc -r reaches.json -d /data/input --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar --latlon /data/input/ssc/training_data_lon_lat.json

# ec2
# sudo docker run -v /mnt/input/:/data/input -v /home/ec2-user/:/root/ ssc_deploy -j /data/input/ssc/hls_links_na.json -o /data/input/ -r reaches.json -d /data/input --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar --latlon /data/input/ssc/training_data_lon_lat.json -i 10
# sudo docker run -v /mnt/input/:/data/input -v /home/ec2-user/:/root/ ssc_deploy -j /data/input/ssc/2024_oct/HLS_links_saved.json -o /data/input/ -r reaches.json -d /data/input --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar --latlon /data/input/ssc/2024_oct/coords_json.json -i 10

# sudo docker run -v /home/ec2-user/temp_data/:/data/input -v /home/ec2-user/:/root/ ssc_deploy \
#     -j /data/input/ssc/2025_jan/HLS_links_noorder_saved.json \
#         -o /data/input/ \
#             -r reaches.json \
#                 -d /data/input \
#                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar \
#                         --latlon /data/input/ssc/2025_jan/coords_noorder_json.json \
#                             -i 10

# singularity run --bind /nas/cee-water/cjgleason/travis/repos/singularity_sifs/:/data/input,/home/ec2-user/:/root/ ssc_deploy.simg \
#     -j /data/input/ssc/2025_jan/HLS_links_noorder_saved.json \
#         -o /data/input/ \
#             -r reaches.json \
#                 -d /data/input \
#                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar \
#                         --latlon /data/input/ssc/2025_jan/coords_noorder_json.json \
#                             -i 10

# singularity run --bind /nas/cee-water/cjgleason/travis/repos/singularity_sifs/:/data/input,/home/tsimmons_umass_edu:/root/ ssc_deploy.simg \
#     -j /data/input/ssc/2025_jan/HLS_links_noorder_saved.json \
#         -o /data/input/ \
#             -r reaches.json \
#                 -d /data/input \
#                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar \
#                         --latlon /data/input/ssc/2025_jan/coords_noorder_json.json \
#                             -i 10
## From HPC
# singularity run --bind /nas/cee-water/cjgleason/travis/repos/singularity_sifs/:/data/input,/home/tsimmons_umass_edu:/root/ ssc_deploy.simg     -j /data/input/ssc/2025_jan/HLS_links_noorder_saved.json         -o /data/input/             -r reaches.json                 -d /data/input                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar                         --latlon /data/input/ssc/2025_jan/coords_noorder_json.json                             -i 10

## Verify
# sudo docker run -v /mnt/input:/data/input -v /home/ec2-user/:/root/ ssc_deploy \
#     -j /data/input/ssc_mar_18/ssc_verify/HLS_links_input.json \
#         -o /data/input/ssc_mar_18/results \
#             -r reaches.json \
#                 -d /data/input \
#                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar \
#                         --latlon /data/input/ssc_mar_18/ssc_verify/coords_json_input.json \
#                           --ann_model_dir /data/input/ssc/models/model_static_files/ \
#                             -i 0


# # Verify june26
# sudo docker run -v /mnt/input:/data/input -v /home/ec2-user/:/root/ ssc_deploy \
#     -j /data/input/global_HLS_filtered_plus_minus_1_day/af_hls_list_chunk_5_time_343_plus_minus_one_day.json \
#         -o /data/input/ssc_june_26/results \
#             -r reaches.json \
#                 -d /data/input \
#                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar \
#                           --ann_model_dir /data/input/ssc/models/model_static_files/ \
#                             -i 0


# sudo docker run -v /mnt/input:/data/input -v /home/ec2-user/:/root/ ssc_deploy     -j /data/input/global_HLS_filtered_plus_minus_1_day_compressed_no_dupes  
    #    -o /data/input/ssc_june_26/results                  -d /data/input                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar                           --ann_model_dir /data/input/ssc/models/model_static_files/final_model_static_files                             -i 2 -c > output.log 2>&1

# sudo docker run -v /mnt/validation/stats:/data/validation -v /mnt/input:/data/input -v /home/ec2-user/:/root/ ssc_deploy    --validation_dir /data/validation  -j /data/input/global_HLS_filtered_plus_minus_1_day_compressed_no_dupes         -o /data/input/ssc_june_26/results                  -d /data/input                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar                           --ann_model_dir /data/input/ssc/models/model_static_files/final_model_static_files                             -i 2 -c > output.log 2>&1

# Standard imports
import datetime
import argparse
import os
import time
import glob
import json
import sys
import logging
from collections import defaultdict

# Local imports
import ssc.input
# import ssc.preprocessing
from ssc.generate_feats_multitask import multitask_model_deploy, load_multitask_model
from ssc.crop_bands import crop_bands
#from ssc.ann_ssc_model_v2 import ann_ssc_model #commenting for clarity
from ssc.output import feature_output
# import cv_preprocessing, ssc_preprocessing
# from ssc.multitasking_vision_model import multitasking_vision_model
from ssc.ann_ssc_model import ann_ssc_model
from ssc.ann_ssc_model import load_ann_scc_model
from ssc.calculate_sedflux import calculate_sedflux
# from ssc.output import output

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

def create_args():
    """Create and return argparser with arguments."""

    arg_parser = argparse.ArgumentParser(description="Retrieve a list of S3 URIs")

    arg_parser.add_argument("-i",
                            "--index",
                            type=int,
                            help="Index value to select tile to run on")

    arg_parser.add_argument("-j",
                            "--hls_s3_json_filename",
                            type=str,
                            default='foo',
                            help="Filename of JSON file with list S3 paths to HLS tiles, or the directory for HLS files if chunking")
    
    arg_parser.add_argument("-d",
                            "--indir",
                            type=str,
                            help="input directory",)

    arg_parser.add_argument("-o",
                            "--outdir",
                            type=str,
                            help="Output directory",)
    
    arg_parser.add_argument("-r",
                            "--reaches_of_interest_path",
                            type=str,
                            help="Filepath to json containing reaches of interest",
                            default=None)
    
    arg_parser.add_argument("--max_thresh",
                            type=float,
                            default=0.2,
                            help="Threshold for water mask prediction, can be in range of [0,1] - default is 0.2")
    
    arg_parser.add_argument("--save_masks_bool",
                            type=bool,
                            default=False,
                            help="Set to True if masks should be saved for visual checks")
    
    arg_parser.add_argument('--ckpt_path',
                            default='ckpts/deeplabv3p_distrib.pth.tar',
                            type=str,
                            help='Checkpoint path of multitask model to use')
    
    arg_parser.add_argument('--backbone',
                            default='deeplabv3p', 
                            type=str,
                            help='Backbone of model being used')

    arg_parser.add_argument('--is_distrib', 
                            default=1, 
                            type=int,
                            help='Flag to indicate if model was trained in distributed way (1=distributed, 0=not)')
    
    arg_parser.add_argument('--buffersize', 
                            default=300, 
                            type=int,
                            help='How many meters to crop around the nodes')
    
    arg_parser.add_argument('--latlon', 
                            type=str,
                            help='If you want to use a specific lat lon instead of searching for reaches')

    arg_parser.add_argument('-t',
                            '--training', 
                            action='store_true',
                            help='Indicates that you want to save the training data')
    
    arg_parser.add_argument('--ann_model_dir', 
                            type=str,
                            help='directory of ann model',
                            default= '/mnt/input/ssc/models/model_static_files/nd_20250430'),

    arg_parser.add_argument('--run_location', 
                            type=str,
                            help='Indicates where we are running to define the tile link prefix',
                            default= 'aws'),    
    
    arg_parser.add_argument('--consensus_dir', 
                            type=str,
                            help='Path to the consensus files which hold the FLPE consesus discharge for calculating sedflux'),
    
    arg_parser.add_argument('-c',
                            '--chunk_processing', 
                            action='store_true',
                            help='Passing this arg allows you to process whole files instead of one file parallel by tile')

    return arg_parser

def main():
    """Main function to execute ssc prediction operations."""

    
    start = datetime.datetime.now()
    
    # Command line arguments
    arg_parser = create_args()
    args = arg_parser.parse_args()

    for arg, val in args.__dict__.items():
        logging.info("%s: %s", arg, val)

    # parsing args
    index_to_run = args.index
    max_thresh_for_multitasking_vision_model_preprocessing = args.max_thresh
    save_masks_bool = args.save_masks_bool
    indir = args.indir
    hls_s3_json_filename = args.hls_s3_json_filename
    ckpt_path = args.ckpt_path
    backbone = args.backbone
    is_distrib = args.is_distrib
    buffersize = args.buffersize
    latlon_file = args.latlon
    out_dir = args.outdir
    chunk_processing = args.chunk_processing
    run_location = args.run_location
    ann_model_dir = args.ann_model_dir
    reaches_of_interest_path = args.reaches_of_interest_path
    consensus_dir = args.consensus_dir
    # logging.info('all_files')
    all_files = glob.glob(os.path.join(indir, '*'))
    ssc_files = glob.glob(os.path.join(indir, 'ssc', '*'))
    # logging.info(all_files)

    fail_log = []

    #static paths
    sentinel_shapefile_filepath = os.path.join(indir, 'ssc/Sentinel-2-Shapefile-Index/sentinel_2_index_shapefile.shp')
    # if not os.path.exists(sentinel_shapefile_filepath):
    #     raise ImportError("No Sentinel Shapefile Found...")

    if index_to_run == -235:
        index_to_run = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
        
    if chunk_processing:
        hls_s3_json_filename = sorted(glob.glob(os.path.join(hls_s3_json_filename, '*')))[index_to_run]
        # hls_s3_json_filename = os.path.basename(hls_s3_json_filepath)
        with open(hls_s3_json_filename) as f:
            json_data = json.load(f)
        index_list = list(range(len(json_data)))
    else:
        index_list = [index_to_run]

    ann_model_list_loaded = load_ann_ssc_model(model_dir = ann_model_dir)

    multitask_model_list_loaded = load_multitask_model(ckpt_path=ckpt_path, 
                                                    backbone=backbone, 
                                                    is_distrib=is_distrib)


    for current_index in index_list:
        # Input
        logging.info(f'Running input... on index {current_index}')
        try:
            # all_bands_in_memory, node_ids_reach_ids_lat_lons, tile_filename, l_or_s, tile_code, cloud_cover, date
            all_bands_in_memory, node_ids_reach_ids_lat_lons, tile_filename, l_or_s, tile_code, cloud_cover, date = ssc.input.input(indir=indir,
                                                                index_to_run=current_index, 
                                                                hls_s3_json_filename=hls_s3_json_filename,
                                                                sentinel_shapefile_filepath=sentinel_shapefile_filepath, 
                                                                latlon_file=latlon_file,
                                                                run_location=run_location,
                                                                reaches_of_interest_path = reaches_of_interest_path)

            # if latlon_file is not None:


            #     with open(latlon_file) as json_file:
            #         node_ids_reach_ids_lat_lons = json.load(json_file)
                    
            #     latlon_file_provided = True
            # else:
            #     latlon_file_provided = False
            
            # logging.info('is provided', latlon_file_provided)
            
            logging.info(f'Input Complete {tile_filename}')
            cnt = 0
            feature_dict = defaultdict(list)

            if len(node_ids_reach_ids_lat_lons) == 0:
                logging.info('NO NODES FOUND IN TILE, EXITING...')
                raise ValueError('NO NODES FOUND IN TILE, EXITING...')
            else:
                logging.info('Running ssc prediction on %s nodes.', len(node_ids_reach_ids_lat_lons))


            
            # if latlon_file_provided:
            #     all_node_data = [(i.split(',')[1],i.split(',')[0]) for i in node_ids_reach_ids_lat_lons]
            # else:
            #     pointlat = node_data[2][1]
            #     pointlon = node_data[2][0]
            
            #     all_node_data = 
        
            all_lats = [i[2][1] for i in node_ids_reach_ids_lat_lons]
            all_lons = [i[2][0] for i in node_ids_reach_ids_lat_lons]
            all_mgrs_flags = []
                
            for node_data in node_ids_reach_ids_lat_lons:
                logging.info('processing node %s of %s', cnt, len(node_ids_reach_ids_lat_lons))
                cnt += 1
                # if latlon_file_provided:
                #     node_data = node_data.split(',')
                #     node_data = ['foo', 'foo', (node_data[1], node_data[0])]
                    
                # logging.info(node_data, 'node_data')
                pointlat = node_data[2][1]
                pointlon = node_data[2][0]
                node_id = node_data[0]
                reach_id = node_data[1]

                # Crop all bands to load them into memory for prediction
                cropped_bands_in_memory = crop_bands(all_bands_in_memory = all_bands_in_memory,
                                                                            node_data = node_data,
                                                                            filename = tile_filename, 
                                                                            buffersize = buffersize,
                                                                            l_or_s = l_or_s)
                
                if len(cropped_bands_in_memory) >= 1:

                    # try:
                    #     logging.info('Trying to print cropped band from main script. Here is the cropped band')
                    #     # logging.info(cropped_bands_in_memory[0])
                    # except:
                    #     logging.info('failed to print...')


                    # Multitasking model feature generation
                    try:
                        features_for_ann_model = multitask_model_deploy(all_bands_in_memory=cropped_bands_in_memory,
                                                                    node_data = node_data,
                                                                    ckpt_path=ckpt_path, 
                                                                    backbone=backbone, 
                                                                    is_distrib=is_distrib,
                                                                    l_or_s = l_or_s)

                        features_for_ann_model['reach_id'] = reach_id
                        features_for_ann_model['node_id'] = node_id
                        
                        mgrs_flag = ssc.input.mgrs_flag_generation(tile_code, node_data)
                        all_mgrs_flags.append(mgrs_flag)

                    except Exception as e:
                        logging.info('Node failed...')
                        logging.info(e)
                        continue 

                    for feature, value in features_for_ann_model.items():
                        if value is None:
                            feature_dict[feature].append(-9999)
                        elif isinstance(value, list):
                            feature_dict[feature].extend(value)  # or append if you want it nested
                        else:
                            feature_dict[feature].append(value)




            # logging.info(feature_dict)

            # logging.info(feature_dict.values())
            
            # def feature_output(feature_dict, out_dir, cloud_cover, mgrs_flag, date, node_data):
            # logging.info('HERE IS ANOTHER LS CHECK', l_or_s)

            preprocessed_data_df = feature_output(feature_dict=feature_dict, out_dir = out_dir, cloud_cover = cloud_cover, \
                mgrs_flag = all_mgrs_flags, date = date, l_or_s = l_or_s, args=args, lat = all_lats, lon = all_lons, filename = tile_filename)


            logging.info(preprocessed_data_df['LorS'], 'lorssss ???')
            model_outputs_df = ann_ssc_model(df_hlsprocessed_raw = preprocessed_data_df, ann_model_list_loaded = ann_model_list_loaded)
            logging.info(model_outputs_df['LorS'], 'lorssss bad here')
            
            # logging.info('prediction %s', model_outputs)

            # ssc_preprocessing(indir = indir,
            #                 save_masks_bool=save_masks_bool,
            #                 max_thresh_for_multitasking_vision_model_preprocessing = max_thresh_for_multitasking_vision_model_preprocessing)

            # Output
            # model_outputs_df.to_csv(os.path.join(out_dir, os.path.basename(tile_filename'testing_ann.csv'))
            
            
            logging.info('Calculating sedflux...')

            final_outputs_for_tile = calculate_sedflux(model_outputs_df  = model_outputs_df, consensus_dir = consensus_dir)

            logging.info('Saving multitask model outputs...')
            final_outputs_for_tile.to_csv(os.path.join(out_dir,tile_filename.replace('.tar','') + '.csv'))
            print(os.path.join(out_dir,tile_filename.replace('.tar','') + '.csv'), model_outputs_df)
        
        except Exception as e:
            print('failed...', current_index, e)
            fail_log.append(['failed...', str(current_index), str(e)])
            # with open("/data/input/ssc_june_26/results/fail_log.json", "w") as f:
            #     for row in fail_log:
            #         f.write(json.dumps(row) + "\n")



        end = datetime.datetime.now()
        logging.info(f"Execution time: %s", end - start)

    print('-------------------FAIL LOG HERE')
    print(fail_log)
if __name__ == "__main__":
    main()