# YOLO

# Install (the same for v7 and v8)

```
conda create --name yolo7gpu python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Yolo v7

### Detection

https://github.com/pHidayatullah/yolov7

--source 0, is webcam  
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

### Learning

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
| --epochs       | The number of times the learning algorithm will| --epochs 300                      |
|                | work to process the entire dataset.            |                                    |


```
python train.py --workers 0 --batch-size 4 --device 0 --data /mnt/LinuxFiles/Study/computer_vision/yolo/datasets/face_mask_dataset/face_mask.yaml --img 640 640 --cfg /mnt/LinuxFiles/Study/computer_vision/yolo/datasets/face_mask_dataset/yolov7-face_mask.yaml --weights yolov7_training.pt --name yolov7-face-mask --hyp data/hyp.scratch.custom.yaml --epochs 300
```

## Yolo v8

https://github.com/ultralytics/ultralytics

--source="0", is webcam
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

## Split database

Images should already have the labels for YOLO with the images in the directory.
```
python split_dataset.py --folder ./datasets/face_mask --train 80 --validation 10 --test 10 --dest ./datasets/face_mask_dataset
```


