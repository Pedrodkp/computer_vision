# YOLO

# Install (the same for v7 and v8)

```
conda create --name yolo7gpu python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

This --source 0 or --source="0", is webcam in detection context, but GPU index in learning context.

## Yolo v7

https://github.com/pHidayatullah/yolov7

### Detection

weights are download from: https://github.com/pHidayatullah/yolov7
```
python detect.py --weights yolov7.pt --source inference/images/horses.jpg

python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source inference/images --view-img --save-txt

python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source inference/road.mp4 --view-img --save-txt

python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source 0
```

| Argument     | Description                   | Default           | Example                 |
|--------------|-------------------------------|-------------------|-------------------------|
| --weights    | Weights YOLO v7               | yolov7.pt         | --weights yolov7.pt    |
| --conf-thres | Object confidence threshold   | 0.25              | --conf-thres 0.25       |
| --img-size   | Inference Size                | 640               | --img-size 640          |
| --source     | Images/Videos/Webcam          | “inference/images” | --source inference/images |
| --view-img   | Display Results               | -                 | --view-img              |
| --save-txt   | Save detection result to file | -                 | --save-txt              |

#### Learning Detection

1. Split database with the scripts split_dataset.py
2. Make a file from > https://github.com/pHidayatullah/yolov7/blob/main/data/coco.yaml
3. Make a file from > https://github.com/pHidayatullah/yolov7/blob/main/cfg/training/yolov7.yaml
4. Download file "yolov7_training.pt" from: https://github.com/pHidayatullah/yolov7
5. Trainning:

| Argument       | Description                                     | Example                            |
|----------------|-------------------------------------------------|------------------------------------|
| --workers      | The number of processes that generate batches   | --workers 0                        |
|                | in parallel.                                    |                                    |
| --batch-size   | The number of images processed before updating  | --batch-size 640                  |
|                | the model.                                      |                                    |
| --device       | CUDA device                                     | --device 0                         |
| --data         | Data file                                       | --data data\face_mask.yaml        |
| --img          | Image Size                                      | --img 512 512                     |
| --cfg          | Configuration File                              | --cfg cfg\training\yolov7-face_mask.yaml |
| --weights      | Initial Weights                                 | --weights yolov7_training.pt      |
| --name         | Model Name                                      | --name yolov7-face-mask           |
| --hyp          | Hyperparameter                                  | --hyp data\hyp.scratch.custom.yaml |
| --epochs       | The number of times the learning algorithm will work to process the entire dataset. | --epochs 300                      |


```
python train.py --workers 0 --batch-size 4 --device 0 --data /mnt/LinuxFiles/Study/computer_vision/yolo/datasets/face_mask_dataset/face_mask.yaml --img 640 640 --cfg /mnt/LinuxFiles/Study/computer_vision/yolo/datasets/face_mask_dataset/yolov7-face_mask.yaml --weights yolov7_training.pt --name yolov7-face-mask --hyp data/hyp.scratch.custom.yaml --epochs 300
```

#### Accuracy Detection

| Argument     | Description                      | Example                     |
|--------------|----------------------------------|-----------------------------|
| --weights    | YOLOv7 Weights                   | --weights best.pt           |
| --batch-size | The number of images processed at one time | --batch-size 4        |
| --device     | CUDA device                      | --device 0                  |
| --data       | Data file                        | --data data\face_mask.yaml |
| --img        | Image Size                       | --img 640                   |
| --conf-thres | Object confidence threshold      | --conf-thres 0.01           |
| --iou        | IOU threshold                    | --iou 0.65                  |
| --name       | Folder Name                      | --name yolov7-face-mask-val |
| --task       | Task                             | --task val                  |


```
python test.py --weights runs/train/yolov7-face-mask2/weights/best.pt --batch-size 2 --device 0 --data data/face_mask.yaml --img 640 --conf-thres 0.01 --iou 0.5 --name yolov7-face-mask-val --task val

python test.py --weights runs/train/yolov7-face-mask2/weights/best.pt --batch-size 2 --device 0 --data data/face_mask.yaml --img 640 --conf-thres 0.01 --iou 0.5 --name yolov7-face-mask-test --task test
```

### Classify

Don't care to user v7 for this, because is not native. Prefer use v8.

## Yolo v8 (ultralytics)

https://github.com/ultralytics/ultralytics

### Detection

```
yolo detect predict model=yolov8l.pt source='https://ultralytics.com/images/bus.jpg'

yolo detect predict model=yolov8l.pt source="inference/images/horses.jpg" save=True conf=0.5 show=True

yolo detect predict model=yolov8l.pt source="inference/images" save=True conf=0.5 show=True line_thickness=1

yolo detect predict model=yolov8l.pt source="inference/road.mp4" save=True conf=0.5 show=True

yolo detect predict model=yolov8l.pt source="0" save=True show=True
```

| Argument         | Description                     | Default                | Example                    |
|------------------|---------------------------------|------------------------|----------------------------|
| model            | YOLOv8                          | -                      | model=yolov8l.pt          |
| source           | Source directory for images or videos | “ultralytics/assets” | source=“ultralytics/assets” |
| save             | Save detection result           | False                  | save=True                  |
| conf             | Object confidence threshold     | 0.25                   | conf=0.25                  |
| imgsz            | Inference Size                  | 640                    | imgsz=640                  |
| show             | Display Results                 | False                  | show=True                  |
| save_txt         | Save detection result to file   | False                  | save_txt=True              |
| line_thickness   | Bounding boxes thickness        | 3                      | line_thickness=3           |

#### Learning Detection

1. Split database with the scripts split_dataset.py
2. Make a file like bellow and add to ultralytics/data:
```
# Dataset Path
path: /mnt/LinuxFiles/Study/computer_vision/yolo/datasets/face_mask_dataset

train: images/train # relative to path
val: images/val # relative to path
test: images/test # relative to path

# Class Names
names: 
    0: "Mask"
    1: "No Mask"
    2: "Bad Mask"
```
3. Trainning:

| Argument     | Description                                       | Default       | Example                |
|--------------|---------------------------------------------------|---------------|------------------------|
| model        | The model that we want to use                    | -             | model=yolov8l.pt      |
| data         | Data file                                         | -             | data=data/face_mask.yaml |
| imgsz        | Image Size                                        | 640           | imgsz=640              |
| workers      | The number of processes that generate batches in parallel | 8         | workers=8              |
| device       | Device to run training                            | -             | device=0, device=cpu   |
| batch        | The number of images processed before updating the model | 16        | batch=16               |
| epochs       | The number of times the learning algorithm will work to process the entire dataset | 100 | epochs=100           |
| patience     | Epochs to wait for no observable improvement for early stopping of training | 50 | patience=50          |

```
yolo detect train model=yolov8l.pt data=data/face_mask.yaml imgsz=640 workers=4 batch=8 device=0 epochs=300 patience=50 name=yolov8_face_mask
```

4. Restart (if stop by ctrl+c):
```
yolo detect train model=runs/detect/yolov8_face_mask/weights/last.pt data=data/face_mask.yaml resume=True
```

5. Use:
```
yolo detect predict model=runs/detect/yolov8_face_mask/weights/best.pt source=0
```

#### Accuracy Detection

| Argument     | Description                           | Default        | Example             |
|--------------|---------------------------------------|----------------|---------------------|
| model        | YOLOv8 Model                         | -              | model=yolov8l.pt   |
| data         | Data file                             | -              | data=data/face_mask.yaml |
| device       | Device to run the mAP calculation    | -              | device=0, device=cpu |
| conf         | Object confidence threshold           | 0.001          | conf=0.001          |
| iou          | IOU threshold                         | 0.6            | iou=0.6             |
| name         | Folder name                           | -              | name=face_mask_val  |
| split        | Dataset split to use for mAP calculation | val          | split=val           |

```
yolo detect val model=/mnt/LinuxFiles/Study/computer_vision/yolo/ultralytics/runs/detect/yolov8_face_mask/weights/best.pt data=data/face_mask.yaml device=0 conf=0.001 iou=0.5 name=face_mask_val split=val

yolo detect val model=/mnt/LinuxFiles/Study/computer_vision/yolo/ultralytics/runs/detect/yolov8_face_mask/weights/best.pt data=data/face_mask.yaml device=0 conf=0.001 iou=0.5 name=face_mask_test split=test
```

### Classify

It will describe all elements in image (need classify model).
```
yolo classify predict model=ultralytics/yolov8l-cls.pt source=examples/bee.png save=True

yolo classify predict model=ultralytics/yolov8l-cls.pt source=examples/volleyball.mp4 show=True
```

#### Learning Classify

| Argument     | Description                                         | Default       | Example                        |
|--------------|-----------------------------------------------------|---------------|--------------------------------|
| model        | The model that we want to use                      | -             | model=yolov8l-cls.pt          |
| data         | Dataset path                                       | -             | data=data/classify/weather_dataset |
| imgsz        | Image Size                                         | 224           | imgsz=224                      |
| workers      | The number of processes that generate batches in parallel | 8       | workers=8                      |
| device       | Device to run training                             | -             | device=0, device=cpu           |
| batch        | The number of images processed before updating the model | 16      | batch=16                       |
| epochs       | The number of times the learning algorithm will work to process the entire dataset | 100 | epochs=100                   |
| patience     | Epochs to wait for no observable improvement for early stopping of training | 50 | patience=50                  |
| name         | Folder Name                                        | -             | name=yolov8_weather           |
| Argument | Description            | Default | Example                                         |
| data     | Dataset path           | -       | data=D:\yolov8-gpu\data\classify\weather_dataset |
| imgsz    | Image Size             | 224     | imgsz=224                                       |

```
yolo classify train model=ultralytics/yolov8l-cls.pt data=datasets/weather_dataset imgsz=224 device=0 workers=2 batch=16 epochs=100 patience=50 name=yolov8_weather_classification
```

If stoped a tranning, can continue this way:
```
yolo classify train model=runs/classify/yolov8_weather_classification/weights/last.pt resume=True
```

Using
```
yolo classify predict model=runs/classify/yolov8_weather_classification/weights/best.pt source=datasets/test/weather.png save=True

yolo classify predict model=runs/classify/yolov8_weather_classification/weights/best.pt source=datasets/test/weather-video.mp4 save=True show=True
```

### Segmentation

It will make the same as detection, but using segmentation like painting the objects (need segmentation model)
```
yolo segment predict model=ultralytics/yolov8l-seg.pt source=datasets/test/pose.png save=True

yolo segment predict model=ultralytics/yolov8l-seg.pt source="0" show=True save=True
```

#### Learning Segmentation

Need create data.yaml like detection. The argments is also the same when creating a detection model.

##### The segmentation model consumes more GPU memory than the detection model, because the model is bigger

```
yolo segment train model=ultralytics/yolov8l-seg.pt data=datasets/coffee_leaf.yaml imgsz=640 workers=0 batch=6 device=0 epochs=300 patience=50 name=yolov8_coffee_leaf
```

If stopped, start is the same way as detection

```
yolo segment train model=/mnt/LinuxFiles/Study/computer_vision/yolo/runs/segment/yolov8_coffee_leaf/weights/last.pt resume=True
```

### Tracking

It will track the objects in videos
```
yolo track model=ultralytics/yolov8l.pt source="datasets/test/road.mp4" show=True
```

#### Tracking + Segmentation

It will track the objects in videos using segmentation model
```
yolo track model=ultralytics/yolov8l-seg.pt source="datasets/test/road.mp4" show=True save=True
```

## Split database

### One class with labelling

Images should already have the labels for YOLO with the images in the directory.
#### Important: The origin dataset, should have the images with the labels in the same folder with as passed at --folder argument.
```
python split_dataset.py --folder ./datasets/face_mask --train 80 --validation 10 --test 10 --dest ./datasets/face_mask_dataset
```

### Multiple classes in folders (only images, no labelling)

The origin dataset, should have the images in folders with represents the classes.
```
python split_dataset_class.py --folder ./datasets/weather --train 80 --validation 10 --test 10 --dest ./datasets/weather_dataset
```

#### Validate images

There is a problem with small bit depths (color depth) when training because there is no eough resolution of color, it should be at least 24 bits.

https://pt.wikipedia.org/wiki/Profundidade_de_cor

So we need remove this images, the script will create a folder unused_image along the directory of its had runned.
```
python check_images.py --src ./datasets/weather_dataset
```

### Visual tensorboard

This can be used for stopped tranning and also finished tranning.  
mAP graph, increase is better.  
loss graph, decrease is better.

```
tensorboard --logdir runs\classify\yolov8_weather_classification
```

### Google Colab

It is the same, but using Google Drive, is good to know:
- Can run python and install dependencies.
- Can use Google Drive é local file system.
- Can run programs in python with export http ports, and have access to it, ex: visual tensorboar works there
- Can show images like Jupyter Notebook (is acctually Jupyter with more power)
- Can user GPU for faster trainning use
- Perfect for prove of concept
