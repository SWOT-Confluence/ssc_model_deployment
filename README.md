# ssc_model_deployment
Module for input, preprocessing, and deployment of the CV and SSC based models

# Neural Networks

This module was based on the Artificial Neural Networks developed by Luisa Vieira Lucchese, available at https://github.com/luisalucchese/ann-ssc-module

## Deployment plan
<!-- 
* Static file of a combined vector of sword reaches by continent
* datagen pulls links of delta of overlapping hls tiles since last sos and stores in a json
* paralelize by hls tile
* predict on all reaches and nodes covered by tile
* store parameters in timeseries files -->


* datagen goes through reaches and finds tiles for each of the lines within date range
* stores them in a input json
* process_ssc reads in this and uses the array index to pick out a tile
* checks tile for overlap with nodes
* predicts on nodes in tile
* writes to tile level efs
* ssc_combine module to coalate the results


sudo docker run -v /mnt/flpe/consensus:/data/consensus -v /mnt/input:/data/input -v /home/ec2-user/:/root/ ssc_deploy    --consensus_dir /data/consensus  -j /data/input/global_HLS_filtered_plus_minus_1_day_compressed_no_dupes         -o /data/input/ssc_july_16/results                  -d /data/input                     --ckpt_path /data/input/ssc/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p_distrib.pth.tar                           --ann_model_dir /data/input/ssc/models/model_static_files/final_model_static_files                             -i 2 -c > output.log 2>&1