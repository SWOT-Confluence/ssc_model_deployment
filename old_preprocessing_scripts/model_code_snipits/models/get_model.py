#
# Authors: Wei-Hong Li

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

from models.lsrapp import ConvBnRelu
from loguru import logger

def get_model(args, tasks_outputs, num_inp_feats=3):
	# Return multi-task learning model or single-task model
	if args.backbone == 'segnet':
		from models.segnet import SegNet
		backbone = SegNet(num_inp_feats=num_inp_feats)
		backbone_channels = 64
	elif args.backbone == "mobilenetv3":
		from models.lsrapp import LRASPP
		backbone_channels = 128
		backbone = LRASPP(trunk="mobilenetv3_small", num_filters=backbone_channels, num_inp_feats=num_inp_feats)
		if "mobilenetv3" not in args.head:
			backbone_channels += 32
	elif args.backbone == "deeplabv3p":
		from models.deeplabv3 import DeepLabV3Plus
		backbone_channels = 256
		backbone = DeepLabV3Plus(decoder_channels=backbone_channels, in_channels=num_inp_feats)
	# TODO: more backbones

	if args.method == 'single-task':
		from models.models import SingleTaskModel
		task = args.task
		head = get_head(args.head, backbone_channels, tasks_outputs[task])
		model = SingleTaskModel(backbone, head, task)
	elif args.method == 'vanilla':
		selected_tasks_outputs = {}
		for task, task_output in tasks_outputs.items():
			if task in args.tasks:
				selected_tasks_outputs[task] = task_output
		from models.models import MultiTaskModel
		logger.debug(f"backbone_channels: {backbone_channels}")
		heads = torch.nn.ModuleDict({task: get_head(args.head, backbone_channels, task_output) for task, task_output in zip(args.tasks, selected_tasks_outputs.values())})
		model = MultiTaskModel(backbone, heads, args.tasks)

	return model

def get_stl_model(args, tasks_outputs, task, backbone_name=None, head_name=None):
	# Return single-task learning models
	backbone_name = backbone_name if backbone_name else args.backbone
	head_name = head_name if head_name else args.head
	if backbone_name == 'segnet':
		from models.segnet import SegNet
		backbone = SegNet()
		backbone_channels = 64
	elif backbone_name == "mobilenetv3":
		from models.lsrapp import LRASPP
		backbone_channels = 128
		backbone = LRASPP(trunk="mobilenetv3_small", num_filters=backbone_channels)
	elif backbone_name == "deeplabv3p":
		from models.deeplabv3 import DeepLabV3Plus
		backbone_channels = 256
		backbone = DeepLabV3Plus(decoder_channels=backbone_channels)
	from models.models import SingleTaskModel
	head = get_head(head_name, backbone_channels, tasks_outputs[task])
	model = SingleTaskModel(backbone, head, task)
	return model

def get_head(head, backbone_channels, task_output):
	""" Return the decoder head """
	if head == 'segnet_head':
		from models.segnet import SegNet_head
		return SegNet_head(backbone_channels, task_output)
	elif head == "mobilenetv3_head":
		return MobileNetv3_head(backbone_channels, task_output)
	elif head == "deeplabv3p_head":
		from segmentation_models_pytorch.base import SegmentationHead
		return SegmentationHead(
            in_channels=backbone_channels,
            out_channels=task_output,
            activation=None,
            kernel_size=1,
            upsampling=1,	# no upsampling
        )

class MobileNetv3_head(nn.Module):
	def __init__(self, num_filters, output_channel):
		super(MobileNetv3_head, self).__init__()
		self.pred_task = nn.Sequential(
			ConvBnRelu(num_filters + 32, num_filters, kernel_size=1),
			nn.Conv2d(num_filters, output_channel, kernel_size=1),
		)
	def forward(self, x):
		return self.pred_task(x)

