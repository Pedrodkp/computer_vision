# YOLO

# Install (the same for v7, v8 and v9)

```
conda create --name yolo7gpu python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

This --source 0 or --source="0", is webcam in detection context, but GPU index in learning context.

## Yolo v7

[readme_v7.md](./readme_v7.md)

## Yolo v8 (ultralytics)

[readme_v8.md](./readme_v8.md)

## Yolo v9

[readme_v9.md](./readme_v9.md)

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
- Can run python and install dependencies
- Can use Google Drive é local file system
- Has it's own file system navigation, and simple editor (for edit .yaml for example)
- Can run programs in python with export http ports, and have access to it, ex: visual tensorboar works there
- Can show images like Jupyter Notebook (is acctually Jupyter with more power)
- Can user GPU for faster trainning use
- Perfect for prove of concept
- Will lose all data after close a colab, but can use things like restart tranning if not closed