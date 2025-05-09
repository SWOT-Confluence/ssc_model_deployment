import torch
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 
import os
from loguru import logger
import rasterio
from matplotlib import pyplot as plt
import argparse
import time

from ssc.multitasking_vision_model_functions.get_model import get_model



def get_band_fp_data(data_dir, site_id, date_str):
    fp_dir = os.path.join(data_dir, site_id, date_str, "cropped_imgs") 
    fp_dir_files = os.listdir(fp_dir)
    fn_samp = fp_dir_files[0]
    fn_prefix = fn_samp.split("_")[:-1]
    fn_prefix = "_".join(fn_prefix)
    return fp_dir, fn_prefix

def get_features(band, final_water_mask, nodata_val=-9999):
    # fp_dir should be the actual data######################

    # if band == "8a":
    #     fp = os.path.join(fp_dir, f"{fn_prefix}_15.tif")    # band 15 in cropped_imgs is band8a
    # else:
    #     fp = os.path.join(fp_dir, f"{fn_prefix}_{band:02d}.tif")

    # if os.path.exists(fp):
    #     dataset = rasterio.open(fp)
    #     data = dataset.read(1)
    data = band
    assert data.shape == (512,512)

    data = np.where(data==nodata_val, np.nan, data) # replace nodata vals with nan
    mask = np.where(final_water_mask==1, 1, np.nan) # change masked out values to nan
    
    # Only get 300m buffer around center (20 by 20 pixels)
    mid_x, mid_y = 512//2, 512//2
    dataview = data[mid_x-10:mid_x+10, mid_y-10:mid_y+10]
    mask = mask[mid_x-10:mid_x+10, mid_y-10:mid_y+10]
    assert dataview.shape == (20,20)
    assert mask.shape == (20,20)
    # Implementing round buffer
    #mask[:] = np.nan
    maskcir=mask.copy()
    maskcir[:] = np.nan
    radius_mask = 10 # 10 pixels - only in UTM projected dataset
    for i in range(0,radius_mask*2+1):
        for j in range(0,radius_mask*2+1):
            #print(i)
            xmap=i-10+0.5
            ymap=j-10+0.5
            radius_inner=xmap*xmap+ymap*ymap
            #print(radius_inner)
            if radius_inner < radius_mask*radius_mask:
                maskcir[i,j]=1.0
            
            

    masked_feat = dataview * mask * maskcir
    band_min = np.nanmin(masked_feat)
    band_mean = np.nanmean(masked_feat)
    band_max = np.nanmax(masked_feat)
    band_std = np.nanstd(masked_feat)
    band_median = np.nanmedian(masked_feat)
    tmp = np.where(masked_feat==0, np.nan, masked_feat)   # treat nan values as zero
    tmp = np.where(np.isnan(tmp), 0, tmp)
    band_count = np.count_nonzero(tmp)

    # stats for whole band, unmasked
    wband_min = np.nanmin(data)
    wband_mean = np.nanmean(data)
    wband_max = np.nanmax(data)
    wband_std = np.nanstd(data)
    wband_median = np.nanmedian(data)
        # band_count = np.count_nonzero(masked_feat)
    # else:
    #     logger.warning(f"{fp} does not exist")
    #     band_min = None
    #     band_mean = None
    #     band_max = None
    #     band_std = None
    #     band_median = None
    #     band_count = None
    return band_min, band_mean, band_max, band_std, band_median, band_count, wband_min, wband_mean, wband_max, wband_std, wband_median

def get_input_data(fp_dir, fn_prefix):   # uses cropped tiles (512 x 512) as input to the model (center of image is ssc sample)
    split = "test"
    feat_bands = ["02", "03", "04", "15", "11", "12"]   # 15 is 8A
    
    feat_bands_data = []
    for band in feat_bands:
        fp = os.path.join(fp_dir, f"{fn_prefix}_{band}.tif")
        dataset = rasterio.open(fp)
        data = dataset.read(1)
        assert data.shape == (512,512)
        feat_bands_data.append(data)
    features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis

    return features

def prepare_inp_data(features, nodata_val=-9999):
    feats_tmp = np.where(features==nodata_val, np.nanmax(features), features)
    feats_min = np.nanmin(feats_tmp)
    feats_tmp = np.where(features==nodata_val, feats_min, features)    # replace nodata values with minimum
    feats_tmp = np.where(np.isnan(feats_tmp), feats_min, feats_tmp)    # replace nan values with minimum
    feats_tmp = np.nan_to_num(feats_tmp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = feats_tmp.astype(np.float32)
    image = data_transforms(image)
    # Normalize data
    if (torch.max(image)-torch.min(image)):
        image = image - torch.min(image)
        image = image / torch.maximum(torch.max(image),torch.tensor(1))
    else:
        image = np.zeros_like(image)
    try:
        image = image.type(torch.FloatTensor)
    except:
        image = torch.from_numpy(image)
    return image

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
def get_masks(all_bands_in_memory, ckpt_path, backbone, is_distrib=True):
    inp_data = all_bands_in_memory
    opt = Namespace(
        ckpt_path=ckpt_path,
        backbone=f'{backbone}',
        head=f'{backbone}_head',
        method='vanilla', 
        tasks=['water_mask', 'cloudshadow_mask', 'cloud_mask', 'snowice_mask', 'sun_mask'], 
    )
    num_inp_feats = 6   # number of channels in input
    tasks_outputs = {
        "water_mask": 1,
        "cloudshadow_mask": 1,
        "cloud_mask": 1,
        "snowice_mask": 1,
        "sun_mask": 1,
    }
    model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats)

    logger.debug(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if is_distrib:
        new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
        checkpoint["state_dict"] = new_ckpt
    tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.debug(f"After loading ckpt: {tmp}")
    logger.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")
    if backbone == "deeplabv3p":
        optim_threshes = {     # for DeepLabv3+
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.3,    # for 261k dataset
        }
    elif backbone == "mobilenetv3":
        optim_threshes = {     # for MobileNet
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.3,
        }
    elif backbone == "segnet":
        optim_threshes = {     # for SegNet
            "water_mask": 0.2,
            "cloudshadow_mask": 0.2,
            "cloud_mask": 0.3,
            "snowice_mask": 0.2,
            'sun_mask': 0.5,
        }
    model.eval()
    # inp_data = np.stack(inp_data, axis=-1)
    # print(inp_data)
    # for i in range(10):
    #     print(inp_data[0][0:10], 'these should not be the same')
    print('shape of one', inp_data[0].shape) # this should be (512, 512, 6) select the first six
    if inp_data[0].shape != (512,512):
        raise ValueError('Shape of data is not 515, 512')

    # -------------------------------------------------------------------
    inp_data = np.array(inp_data) #ONLY FOR DEVELOPMENT
    # -------------------------------------------------------------------
    
    inp_data = np.transpose(inp_data, (1,2,0))

    
    inp_data = prepare_inp_data(inp_data)   # size: (6, 512,512)
    inp_data = torch.unsqueeze(inp_data, dim=0) # to have batch size of 1
    with torch.no_grad():
        start_time = time.time()
        test_pred, feat = model(inp_data, feat=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print('model ran in ', execution_time)
        
    masks = {}
    for t in tasks_outputs.keys():
        pred_img = test_pred[t][0,:,:].detach().cpu().numpy()
        thresh = optim_threshes[t]
        masks[t] = (pred_img > thresh).astype(int).squeeze()

    return masks

ID_cols = ['SiteID', 'lat', 'lon', 'date', 'cloud_cover', 'tss_value', 'relative_day', "MGRS", "LorS"]


def multitask_model_deploy(all_bands_in_memory, node_data, ckpt_path, backbone, is_distrib, l_or_s):
    # args = parser.parse_args()
    # logger.debug(f"args: {args}")

    # split_csv = f"data/{args.split}_ssc.csv"
    # split_csv = pd.read_csv(split_csv)

    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir, exist_ok=True)
    # out_split_dir = os.path.join(args.out_dir, args.split)
    # if not os.path.exists(out_split_dir):
    #     os.makedirs(out_split_dir, exist_ok=True)


    # split_data = split_csv[ID_cols]
    # row = split_data.iloc[args.i]

    # new_data = row[ID_cols]
    # site_id, date_str = row["SiteID"], row["date"]
    # fp_dir, fn_prefix = get_band_fp_data(args.data_dir, site_id, date_str)   # get directory of tile, and the prefix for cropped imgs

    pointlat = node_data[2][1]
    pointlon = node_data[2][0]

    model_bands = {
        'L':["B02", "B03", "B04", "B05", "B06", "B07"],
        'S':["B02", "B03", "B04", "B8A", "B11", "B12"]
    }

    band_definitions = {
        'S':["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        'L':["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11"]

    }

    assert len(all_bands_in_memory) == len(band_definitions[l_or_s])

    # model_bands_in_memory = [i for i in all_bands_in_memory if i in model_bands[l_or_s]]
    model_bands_in_memory = []
    cnt = 0
    for i in all_bands_in_memory:
        if band_definitions[l_or_s][cnt] in model_bands[l_or_s]:
            model_bands_in_memory.append(i)
        cnt +=1

    assert len(model_bands_in_memory) == len(model_bands[l_or_s])

    print('found the correct number of bands')

    masks = get_masks(all_bands_in_memory=model_bands_in_memory, ckpt_path=ckpt_path, backbone=backbone, is_distrib=(is_distrib==1))

    water_mask = masks["water_mask"]
    cloudshadow_mask = masks["cloudshadow_mask"]
    cloud_mask = masks["cloud_mask"]
    snowice_mask = masks["snowice_mask"]
    sun_mask = masks["sun_mask"]
    final_water_mask = water_mask * (1-cloudshadow_mask) * (1-cloud_mask) * (1-snowice_mask) * sun_mask
    # final_water_mask = water_mask * sun_mask

    	# Unnamed: 0	b1_min	b1_mean	b1_max	b1_std	b1_median	b1_count	b2_min	b2_mean	b2_max	b2_std	b2_median	b2_count	b3_min	
        # b3_mean	b3_max	b3_std	b3_median	b3_count	b4_min	b4_mean	b4_max	b4_std	b4_median	b4_count	b5_min	b5_mean	b5_max	
        # b5_std	b5_median	b5_count	b6_min	b6_mean	b6_max	b6_std	b6_median	b6_count	b7_min	b7_mean	b7_max	b7_std	b7_median	
        # b7_count	b8_min	b8_mean	b8_max	b8_std	b8_median	b8_count	b9_min	b9_mean	b9_max	b9_std	b9_median	b9_count	b10_min	b10_mean	
        # b10_max	b10_std	b10_median	b10_count	b11_min	b11_mean	b11_max	b11_std	b11_median	b11_count	b12_min	b12_mean	b12_max	b12_std	b12_median	
        # b12_count	b13_min	b13_mean	b13_max	b13_std	b13_median	b13_count	b14_min	b14_mean	b14_max	b14_std	b14_median	b14_count	b8a_min	b8a_mean	
        # b8a_max	b8a_std	b8a_median	b8a_count	Fmask_min	Fmask_mean	Fmask_max	Fmask_std	Fmask_median	Fmask_count	SAA_min	SAA_mean	SAA_max	SAA_std	
        # SAA_median	SAA_count	SZA_min	SZA_mean	SZA_max	SZA_std	SZA_median	SZA_count	VAA_min	VAA_mean	VAA_max	VAA_std	VAA_median	VAA_count	VZA_min	VZA_me
        # an	VZA_max	VZA_std	VZA_median	VZA_count	MGRS	LorS	SiteID	lat	lon	date	cloud_cover	tss_value	relative_day
    new_data = {}
    for band in list(range(len(all_bands_in_memory))):
        band_min, band_mean, band_max, band_std, band_median, band_count, wband_min, wband_mean, wband_max, wband_std, wband_median = get_features(
        all_bands_in_memory[band], final_water_mask
        )

        new_data[f"{band_definitions[l_or_s][band]}_min"] = band_min
        new_data[f"{band_definitions[l_or_s][band]}_mean"] = band_mean
        new_data[f"{band_definitions[l_or_s][band]}_max"] = band_max
        new_data[f"{band_definitions[l_or_s][band]}_std"] = band_std
        new_data[f"{band_definitions[l_or_s][band]}_median"] = band_median
        new_data[f"{band_definitions[l_or_s][band]}_count"] = band_count
        new_data[f"{band_definitions[l_or_s][band]}_wmin"] = wband_min
        new_data[f"{band_definitions[l_or_s][band]}_wmean"] = wband_mean
        new_data[f"{band_definitions[l_or_s][band]}_wmax"] = wband_max
        new_data[f"{band_definitions[l_or_s][band]}_wstd"] = wband_std
        new_data[f"{band_definitions[l_or_s][band]}_wmedian"] = wband_median
        new_data['lat'] = pointlat
        new_data['lon'] = pointlon

    # df = pd.DataFrame(new_data.values.reshape(1,-1), columns=list(new_data.keys()))
    # out_fp = os.path.join(out_split_dir, f"{args.i:06d}.csv")
    # df.to_csv(out_fp, index=False)
    return new_data