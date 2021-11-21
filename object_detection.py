import argparse
import numpy as np
import cv2
import random
import cv2
import os

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_train_loader(cls, cfg):
    return build_detection_train_loader(
        cfg, 
        mapper=DatasetMapper(
            cfg, 
            is_train=True, 
            augmentations=[
                           T.RandomFlip(prob=0.5),
                           T.augmentation_impl.RandomBrightness(0.5, 1.5),                           
                           T.RandomLighting(scale=1.4)                           
                           ]
        )
    )

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def parser():
  parser = argparse.ArgumentParser("Object Detection using Facebook Detectron2 Model ")
  parser.add_argument("--config_file", type = str, default = "Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml", help = "Detecton yaml config file")
  parser.add_argument("--weights", type = str, default = "", help = "weights file path")
  parser.add_argument("--input_image", type = str, default = "", help = "input image path")
  parser.add_argument("--num_classes", type = int, default = 0, help = "number of classes")
  parser.add_argument("--dataset_dir", type = str, default = "", help = "Directory of test images")
  parser.add_argument("--annot_dir", type = str, default = "", help = "Annotation path")
  parser.add_argument("--thresh", type=float, default = 0.15, help = "threshold value")
  parser.add_argument("--model_dir", type = str, default = "", help = "Directory where model is stored")
  return parser.parse_args()

def check_parameters(args):
  if args.weights == "":
    raise(ValueError("No weights found"))
  if args.input_image == "":
    raise(ValueError("No input image found"))
  if args.num_classes == 0:
    raise(ValueError("Class count cannot be zero"))

def main():
  args = parser()
  check_parameters(args)

  register_coco_instances("my_dataset_test", {}, args.annot_dir, args.dataset_dir)

  cfg = get_cfg()
  cfg.OUTPUT_DIR = args.model_dir
  cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
  cfg.DATASETS.TRAIN = ("my_dataset_test",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2

  #this is number of clases
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes


  #obtaining trainer
  trainer = CocoTrainer(cfg) 
  cfg.MODEL.WEIGHTS = args.weights
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
  predictor = DefaultPredictor(cfg)
  im = cv2.imread(args.input_image)
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)

  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  #save the prediced image
  cv2.imwrite('output.jpg', out.get_image()[:, :, ::-1])  

if __name__ == "__main__":
  main()