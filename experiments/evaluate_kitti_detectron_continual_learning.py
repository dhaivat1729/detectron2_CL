import random
import os
import glob 
import cv2
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import sys
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from contextlib import redirect_stdout
import argparse
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



class_list_all = ['Car', 'Van', 'Truck', 'Tram']
# class_list = ['Pedestrian', 'Cyclist', 'Person_sitting', 'Tram']

# write a function that loads the dataset into detectron2's standard format
def get_kitti_dicts(root_dir, data_label, class_dict):
    
    image_names = sorted(glob.glob(root_dir+"/images/training/*.png"))
    train_images = int(len(image_names)*0.75)
    test_images = len(image_names) - train_images
    if data_label == 'train':
        image_names = image_names[:train_images]
        # image_names = image_names[0:100]
    if data_label == 'test':
        # import ipdb; ipdb.set_trace()
        image_names = image_names[-test_images:]
    # print(image_names)
    # image_names = image_names[0:10]
        
    record = {}
    dataset_dicts = []

    class_names = list(class_dict.keys())
    print(class_names)
    for idx, name in enumerate(image_names):
        # print(name)
        record = {}
        height, width = cv2.imread(name).shape[:2]
        # import ipdb; ipdb.set_trace()
        record["file_name"] = name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        label_name = root_dir+"/labels/training/"+name[-10:-3]+"txt"
        # print(label_name)
        ob_list = []
        ## Creating a dictionary expected by detectron
        with open(label_name) as file:
            objects = file.read().splitlines()
            objs = []
            for obj in objects:
                obj = obj.split()
                if obj[0] in class_names:
                    obj_ann = {
                        "bbox": [float(i) for i in obj[4:8]],
                        "bbox_mode": BoxMode.XYXY_ABS,                    
                        "category_id": class_dict[obj[0]],
                        "iscrowd": 0
                    }

                    objs.append(obj_ann)
            # print(len(objs))
            record["annotations"] = objs
            # print(record["annotations"])
        dataset_dicts.append(record)

    return dataset_dicts


# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/fine_tuning/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/distillation_loss/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/rehearsal/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/upperbound_data/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/loss_attenuation_sampling_distillation/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/distillation_loss_more_training_iterations/"
# output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/loss_attenuation_distillation_weighted/"
output_main_dir = "/network/tmp1/bhattdha/detectron2CL_kitti/loss_attenuation_box_ratio_distillation/"

class_dict = {}
for ind, class_name in enumerate(class_list_all[:2]):
    class_dict[class_name] = ind

    print(ind, class_name)
    from detectron2.data import DatasetCatalog, MetadataCatalog

    root_dir = '/network/tmp1/bhattdha/kitti_dataset'
    for d in ["train", "test"]:
        DatasetCatalog.register("kitti/" + d+class_name, lambda d=d: get_kitti_dicts(root_dir, d, class_dict))
        MetadataCatalog.get('kitti/' + d+class_name).set(thing_classes=class_list_all)

    # kitti_metadata = MetadataCatalog.get('kitti/train')

    print("data loading")
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg = torch.load(os.path.join(output_main_dir + class_name + '/' + class_name + '_cfg.final'))['cfg']

    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    cfg.DATASETS.TEST = ('kitti/test'+class_name,)   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list_all)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = os.path.join(output_main_dir, class_name)
    cfg.MODEL.WEIGHTS = os.path.join(output_main_dir, class_name, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    if cfg.CUSTOM_OPTIONS.DETECTOR_TYPE is 'deterministic':
        ## has to be smooth l1 loss if detector is deterministc
        cfg.CUSTOM_OPTIONS.LOSS_TYPE_REG = 'smooth_l1'

    print("reaching here")
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator('kitti/test'+class_name, cfg, False, output_dir="./distillation_loss/" + class_name)
    val_loader = build_detection_test_loader(cfg, 'kitti/test'+class_name)
    results  = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)

# """Now, we perform inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:"""

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("kitti/test", )
# predictor = DefaultPredictor(cfg)

# """Then, we randomly select several samples to visualize the prediction results."""

# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_kitti_dicts("/network/tmp1/bhattdha/kitti_dataset", 'test')
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     print(im)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=kitti_metadata, 
#                    scale=1.0, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )

#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
#     # cv2_imshow(v.get_image()[:, :, ::-1])
    
