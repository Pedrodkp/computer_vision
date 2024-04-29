# YOLO

# Install (the same for v7 and v8)

```
conda create --name yolo7gpu python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Yolo v7

https://github.com/pHidayatullah/yolov7

--source 0, is webcam
```
python detect.py --weights yolov7.pt --source inference/images/horses.jpg
python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source inference/images --view-img --save-txt
python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source inference/road.mp4 --view-img --save-txt
python detect.py --weights yolov7.pt --conf-thres 0.5 --img-size 640 --source 0
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
