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