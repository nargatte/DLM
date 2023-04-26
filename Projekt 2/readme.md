# Deep Learning Methods, Project No. 2, Marcin Galiński, Grzegorz Krzysiak

## Requirements
Project was written in Python 3.8.10 using TensorFlow. The simpliest way to run it is to use Docker dev container and VS Code. After opening directory, Code should detect dev container and offer reopening directory in dev container, which has TensorFlow and all its dependencies installed and configured.

Dev container should be able to run the project regardless of host platform, but it might need some preparation in order to have access to GPU, which is not required, but speeds up running a lot.

On Windows 11 and Windows 10 using WSL 2, the only thing needed is Docker Desktop installed in host system (i.e. Windows). On Linux, apart from Docker, one has to install [Nvidia Docker Support](https://github.com/NVIDIA/nvidia-docker). Note, that it supports limited number of Linux distos.

## Running
Project was done in 4 parts:
 - data exploration, done in Jupyter notebooks saved in files `data_exploration.ipynb` and `data_exploration_test.ipynb`
 - stage 1, with tests written in `stage1.py` and results analysis and postprocessing in `stage1_interpretation.ipynb`
 - stage 2, with tests written in `stage2.py` and results analysis and postprocessing in `stage2_interpretation.ipynb`
 - stage 3, with tests written in `stage3.ipynb` and results analysis and postprocessing in `stage3_interpretation.ipynb`

Data exploration was used to implement methods in file `get_dataset.py`, used in every file to get train and test data. This data is cached in `__pycache__` directory. It should be cached if it wasn't already on each retreaval of data, but to be sure one can first invoke `run_this_to_cache_dataset.py` to cache it.

