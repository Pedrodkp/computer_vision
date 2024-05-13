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

### EXTRAs

#### Pose Estimation

```
python detect_pose.py --weights yolov7-w6-pose.pt --source=../datasets/test/pose.png --kpt-label --view-img

python detect_pose.py --weights yoov7-w6-pose.pt --source=../datasets/test/pose-video.mp4 --kpt-label --view-img
```

#### Squat Count

```
python ./squat-counter/squat_counter.py --weights ./yolov7/yolov7-w6-pose.pt --source ~/Downloads/Agachamentos.mp4 --kpt-label --view-img
```
