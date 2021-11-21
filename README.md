# Object detection using detectron <br>
### In this project, I have trained the different detectron models on COCO dataset, and written a script that allows me to inference the model on image and outputs a single prediction file.

### The Detectron2 training can be found <a href = "https://github.com/yushendye/Detectron-Object-Detection/blob/main/detectron2%20training.ipynb"> here </a>

### The file named object_detection.py is used for running inference on images
```
usage of object_detection.py: Object Detection using Facebook Detectron2 Model  [-h]
                                                         [--config_file CONFIG_FILE]
                                                         [--weights WEIGHTS]
                                                         [--input_image INPUT_IMAGE]
                                                         [--num_classes NUM_CLASSES]
                                                         [--dataset_dir DATASET_DIR]
                                                         [--annot_dir ANNOT_DIR]
                                                         [--thresh THRESH]
                                                         [--model_dir MODEL_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Detecton yaml config file
  --weights WEIGHTS     weights file path
  --input_image INPUT_IMAGE
                        input image path
  --num_classes NUM_CLASSES
                        number of classes
  --dataset_dir DATASET_DIR
                        Directory of test images
  --annot_dir ANNOT_DIR
                        Annotation path
  --thresh THRESH       threshold value
  --model_dir MODEL_DIR
                        Directory where model is stored
```

### Usage of the file <br>
```
!python /content/test_detectron.py --weights /content/drive/MyDrive/18_Aug_Detectron/model_final.pth --input_image /content/test_detectron.py --num_classes 6 --dataset_dir /content/test_data/ --annot_dir /content/main.json --model_dir /content/drive/MyDrive/18_Aug_Detectron
```
