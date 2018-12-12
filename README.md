# Deep Watershed Detection
This repository holds the code to train and evaluate DWD models.
The documentation is intended enable easy use of the code, for an explanation of the
methods please refer to the original publication.

#### Installation
The code is tested on python version 3.6 and Tensorflow 1.4.1, to install dependencies run inside the repo base directory (`repo_home`):
`pip install -r requirements.txt`. The image database module taken from the original [fast r-cnn repository](https://github.com/rbgirshick/fast-rcnn) and it has some cython code that needs to be compiled by running `make all` inside lib.

#### Data
The training data is expected to be inside `repo_home/data`, you can set up DeepScores_dense (which should suffice for most applications) by running:
```
cd repo_home/lib/demo
python setup_deepscores.py
```
The full dataset can be downloaded [here](https://tuggeluk.github.io/downloads/). For Pascal VOC data
please refer to the [official website](http://host.robots.ox.ac.uk/pascal/VOC/).

#### Model Training
To train a DWD model DeepScores_dense run:
```
cd repo_home/lib
python main/train_dwd.py
```
`train_dwd.py` contains a variety of configuration possibilities, most of them are easily understood
from the help. Some of the most important ones are explained here:

+ Input Scaling

   ...
+ Pretraining

  ...

+ Definition of training assignments

  ...



#### Model Evaluation
In order to evaluate a model run:

cd DeepWatershedDetection/lib/main
python inference.py

On the main method, give the path of the model we are using to evaluate


#### Recent changes
- a large part of the code is cleaned, made more human readable and partially optimized.
- implements data augmentation (see WORMS paper from Elezi, Tugginer, Pelillo and Stadelmann).
- implements l2-regularization.
- implements random search for hyperparameter optimization.
- in addition to rmsprop, the nets now can be trained using adam and sgd with momentum (we have done hyperparameter optimization only using rmsprop)
- implements focal loss (it has been tested that it works, but no hyperparameter optimization has been done).
- you do not need to track the information on hyperparameters and results, all the needed information is stored on the same folder where is the net.
- run.sh allows sequential running.
- a lot of visualization code, for both debugging and testing purposes
