# NOTE: Run as `python run_multitask_eval.py`
# If sample data is needed, it's available here: https://drive.google.com/drive/folders/1lU2Fcmv1DpiSLc5FZEewFY9gGTyuiyo_?usp=drive_link (place sample_data folder inside model_code_snipits/)
# If checkpoints are needed, download here: https://drive.google.com/drive/folders/1lU2Fcmv1DpiSLc5FZEewFY9gGTyuiyo_?usp=drive_link (place multitask_ckpts folder inside model_static_files/)

# fp_dir = 'https://drive.google.com/drive/folders/1lU2Fcmv1DpiSLc5FZEewFY9gGTyuiyo_?usp=drive_link (place sample_data folder inside model_code_snipits/'
data_dir = 'downloaded cropped images'

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from loguru import logger
import numpy as np
import pandas as pd
import rasterio

import matplotlib.pyplot as plt

from models.get_model import get_model

def get_raw_input_data():   # uses large tiles as input to the model
    # fp_dir = "sample_data/21KAN001_WQX-SC520/2015-12-02"
    # sat_type = "L"
    fp_dir = "/media/confluence/work/repos/confluence/ssc_model_deployment/models/model_code_snipits/sample_data/sample_data/21KAN001_WQX-SC520/2015-12-02"
    sat_type = "L"
    if sat_type == "L": # landsat feat bands
        feat_bands = ["02", "03", "04", "05", "06", "07"]
    else:   # sentinel feat bands
        feat_bands = ["02", "03", "04", "8A", "11", "12"]
    fp_dir_files = os.listdir(fp_dir)
    fn_samp = fp_dir_files[0]
    fn_prefix = fn_samp.split("B")[:-1]
    fn_prefix = "B".join(fn_prefix)
    feat_bands_data = []
    for band in feat_bands:
        fp = os.path.join(fp_dir, f"{fn_prefix}B{band}.tif")
        dataset = rasterio.open(fp)
        data = dataset.read(1)
        feat_bands_data.append(data)
    features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis
    return features

# def get_input_data():   # uses cropped tiles (512 x 512) as input to the model (center of image is ssc sample)
#     split = "test"
#     feat_bands = ["02", "03", "04", "15", "11", "12"]
    
#     data_dir = "/media/confluence/work/repos/confluence/ssc_model_deployment/models/model_code_snipits/sample_data/cropped"  # location of the downloaded tiles
#     date_str, site_id = '2014-09-22', '21FLORAN_WQX-LWB'
#     fp_dir = os.path.join(data_dir, site_id, date_str, "cropped_imgs") 
#     fp_dir_files = os.listdir(fp_dir)
#     fn_samp = fp_dir_files[0]
#     fn_prefix = fn_samp.split("_")[:-1]
#     fn_prefix = "_".join(fn_prefix)
#     feat_bands_data = []
#     for band in feat_bands:
#         fp = os.path.join(fp_dir, f"{fn_prefix}_{band}.tif")
#         dataset = rasterio.open(fp)
#         data = dataset.read(1)
#         assert data.shape == (512,512)
#         feat_bands_data.append(data)
#     features = np.stack(feat_bands_data, axis=-1)   # stack in 3rd axis

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
    image = image.type(torch.FloatTensor)
    return image

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt = Namespace(
    ckpt_path="/media/confluence/work/repos/confluence/ssc_model_deployment/models/model_static_files/rangel_checks/multitask_ckpts/deeplabv3p.pth.tar",
    backbone='deeplabv3p',
    head='deeplabv3p_head',
    method='vanilla', 
    tasks=['water_mask', 'cloudshadow_mask', 'cloud_mask', 'snowice_mask', 'sun_mask'], 
)
# tasks = opt.tasks
num_inp_feats = 6   # number of channels in input
tasks_outputs = {
    "water_mask": 1,
    "cloudshadow_mask": 1,
    "cloud_mask": 1,
    "snowice_mask": 1,
    "sun_mask": 1,
}

model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats)
if os.path.exists(opt.ckpt_path):
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.debug(f"Loading checkpoint from {opt.ckpt_path}. epoch: {checkpoint['epoch']}")
else:
    logger.warning(f"No checkpoint found in {opt.ckpt_path}. Running untrained model.")
model.eval()

# Define optimal thresholds for different model types
if opt.backbone == "deeplabv3p":
    optim_threshes = {     # for DeepLabv3+
        "water_mask": 0.2,
        "cloudshadow_mask": 0.2,
        "cloud_mask": 0.3,
        "snowice_mask": 0.2,
        'sun_mask': 0.2,
    }
elif opt.backbone == "mobilenetv3":
    optim_threshes = {     # for MobileNet
        "water_mask": 0.2,
        "cloudshadow_mask": 0.2,
        "cloud_mask": 0.3,
        "snowice_mask": 0.2,
        'sun_mask': 0.3,
    }
elif opt.backbone == "segnet":
    optim_threshes = {     # for SegNet
        "water_mask": 0.2,
        "cloudshadow_mask": 0.2,
        "cloud_mask": 0.3,
        "snowice_mask": 0.2,
        'sun_mask': 0.5,
    }
logger.debug(f"Using the following thresholds: {optim_threshes}")

is_padded = False
# Use these when using the large tile as input to the model (3660 x 3660)
try:
    inp_data = get_raw_input_data() # size: (3660,3660,6)
    pad_size = 2
    inp_data = np.pad(inp_data, ((pad_size,pad_size),(pad_size,pad_size),(0,0)))    # pad to make size (3664,3664,6)
    is_padded = True
except FileNotFoundError:
    inp_data = np.random.rand(512,512,6)
    logger.warning("No input data found. Using random numbers.")
# inp_data = get_input_data() # size: (512,512,6)   # Use this if want to use the cropped tile with the ssc sample in the middle

logger.debug(f"inp_data: {inp_data.shape}")
inp_data = prepare_inp_data(inp_data)   # size: (6,512,512)

inp_data = torch.unsqueeze(inp_data, dim=0) # to have batch size of 1
with torch.no_grad():
    pred, feat = model(inp_data, feat=True)

# Separate model outputs
water_mask = np.where(pred["water_mask"].detach().numpy() > optim_threshes["water_mask"], 1, 0)
cloudshadow_mask = np.where(pred["cloudshadow_mask"].detach().numpy() > optim_threshes["cloudshadow_mask"], 1, 0)
cloud_mask = np.where(pred["cloud_mask"].detach().numpy() > optim_threshes["cloud_mask"], 1, 0)
snowice_mask = np.where(pred["snowice_mask"].detach().numpy() > optim_threshes["snowice_mask"], 1, 0)
sun_mask = np.where(pred["sun_mask"].detach().numpy() > optim_threshes["sun_mask"], 1, 0)

water_mask = np.squeeze(water_mask) 
cloudshadow_mask = np.squeeze(cloudshadow_mask) 
cloud_mask = np.squeeze(cloud_mask) 
snowice_mask = np.squeeze(snowice_mask) 
sun_mask = np.squeeze(sun_mask) 

if is_padded:
    water_mask = water_mask[pad_size:-pad_size, pad_size:-pad_size]
    cloudshadow_mask = cloudshadow_mask[pad_size:-pad_size, pad_size:-pad_size]
    cloud_mask = cloud_mask[pad_size:-pad_size, pad_size:-pad_size]
    snowice_mask = snowice_mask[pad_size:-pad_size, pad_size:-pad_size]
    sun_mask = sun_mask[pad_size:-pad_size, pad_size:-pad_size]


# Remove cloud shadow, remove cloud, remove snow/ice, and leave water that's illuminated by sun
final_water_mask = water_mask * (1-cloudshadow_mask) * (1-cloud_mask) * (1-snowice_mask) * sun_mask

# Save outputs
if not os.path.exists("out"):
    os.makedirs("out", exist_ok=True)
plt.imshow(water_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("water_mask")
plt.savefig(f"out/water_mask.png", bbox_inches="tight")
plt.close()
plt.imshow(cloudshadow_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("cloudshadow_mask")
plt.savefig(f"out/cloudshadow_mask.png", bbox_inches="tight")
plt.close()
plt.imshow(cloud_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("cloud_mask")
plt.savefig(f"out/cloud_mask.png", bbox_inches="tight")
plt.close()
plt.imshow(snowice_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("snowice_mask")
plt.savefig(f"out/snowice_mask.png", bbox_inches="tight")
plt.close()
plt.imshow(sun_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("sun_mask")
plt.savefig(f"out/sun_mask.png", bbox_inches="tight")
plt.close()
plt.imshow(final_water_mask, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("final_water_mask")
plt.savefig(f"out/final_water_mask.png", bbox_inches="tight")
plt.close()

logger.debug(f"water_mask: {water_mask.shape}")
logger.debug(f"cloudshadow_mask: {cloudshadow_mask.shape}")
logger.debug(f"cloud_mask: {cloud_mask.shape}")
logger.debug(f"snowice_mask: {snowice_mask.shape}")
logger.debug(f"sun_mask: {sun_mask.shape}")
logger.debug(f"final_water_mask: {final_water_mask.shape}")
logger.debug(f"final_water_mask: {np.unique(final_water_mask)}")