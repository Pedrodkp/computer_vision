### Install

The course sugest really old libs, i will try the course with the new ones and upgrade the code by myself as an exercise also.

#### Start: 
- Install mini conda   
```
eval "$(/mnt/LinuxFiles/Apps/conda/bin/conda shell.bash hook)"
```

#### Create env:
```
conda create --name computer_vision python=3.8.18
```

#### Activate env:
```
conda activate computer_vision
```

#### Run env:
```
jupyter-lab
```

#### Deactivate env:
```
conda deactivate
```

#### Install dep (can use pip or conda)
```
conda install jupyter
conda install jupyterlab
conda install keras
conda install matplotlib
conda install opencv
conda install pandas
conda install scikit-learn
conda install scipy
conda install tensorboard
conda install tensorflow
```

#### Export
```
conda env export > environment.yml
```

#### Import (creating)
```
conda env create --name computer_vision -f environment.yml
```

## Images

Shapes are representative by (width, height, channels)  
*Each color channel (red, green, blue) works the same like a gray scale  
