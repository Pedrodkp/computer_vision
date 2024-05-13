## Yolo v9

https://github.com/WongKinYiu/yolov9

### Detection

Args is the same then v8.

```
python ./yolov9/detect.py --weights ./yolov9/yolov9-c-converted.pt --source ./datasets/bus.jpg --conf-thres 0.5 --view-img

python ./yolov9/detect.py --weights ./yolov9/yolov9-c-converted.pt --source https://stunningvisionai.com/assets/football.png

python ./yolov9/detect.py --weights ./yolov9/yolov9-c-converted.pt --source ./test/road.mp4 --view-img

python ./yolov9/detect.py --weights ./yolov9/yolov9-c-converted.pt --source 0 --view-img
```

#### Learning Detection

There is a lot of issues to use v9, is not stable, for use the classes and models from internet ok, but for trainning prefer use v8 because is to much more stable.