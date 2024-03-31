### Install

The course sugest really old libs, i will try the course with the new ones and upgrade the code by myself as an exercise also.

### GPU Support
#### Cuda
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```
#### NVIDIA Driver Instructions (choose one option)
To install the legacy kernel module flavor:
```
sudo apt-get install -y cuda-drivers
```
To install the open kernel module flavor (prefered):
```
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550
```
#### cuDNN
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-11
```
#### Test
Should see both card video and cuba version running:
```
nvidia-smi
```
Also check nvidia cuda toolkit:
```
nvcc -V
```
Also check use in python:
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Start: 
- Install mini conda   
```
eval "$(/mnt/LinuxFiles/Apps/conda/bin/conda shell.bash hook)"
```

#### Find versions:
```
conda search -f keras
```

#### Create env:
```
conda create --name course_cv python=3.9 numpy=1.23.4 jupyter keras=2.12.0 matplotlib opencv pandas scikit-learn scipy
```

#### Activate env:
```
conda activate course_cv
```

#### Run env:
```
jupyter-lab
```

#### Deactivate env:
```
conda deactivate
```

#### Remove env:
```
conda env remove --name course_cv
```

#### Install dep (can use pip or conda)
```
conda install tensorflow-gpu=2.12
conda install -c anaconda cudatoolkit
conda install -c conda-forge cudnn
conda install jupyterlab
```

#### Export
```
conda env export > environment.yml
```

#### Import (creating)
```
conda env create --name course_cv -f environment.yml
```

## Images

Shapes are representative by (width, height, channels)  
*Each color channel (red, green, blue) works the same like a gray scale  
