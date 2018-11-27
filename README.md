# Adaptive Image Sampling using Deep Learning
A PyTorch implementation of the paper:
[Adaptive Image Sampling using Deep Learning and its Application on X-Ray Fluorescence Image Reconstruction](arxiv to be appear)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org/)
- tqdm
```
pip install tqdm
```
- opencv
```
conda install -c conda-forge opencv
```
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
```
pip install tensorboard_logger
```
- h5py
```
conda install h5py
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [ImageNet](http://www.image-net.org/).
Train dataset has 100000 images. Val dataset has 1000 images.
Download the datasets from [here](https://drive.google.com/file/d/1RNfvuZKdf8MZAb1zzVgsFSlX36oc1uPA/view?usp=sharing), 
and then extract it into `$data` directory. Modify the path of `$data` directory in line#48 of file train_NetE.py and line#48 of file train_NetM.py.

### Test Image Dataset
The test image dataset are sampled from [ImageNet](http://www.image-net.org/). It contains 100 images. It is stored in file data_val_100.h5 .

## Usage

### Train

First run
```
python train_NetE.py
```
to train the image inpainting network NetE. 

After NetE is trained, modify the file name of trained NetE in line#29 of file train_NetM.py and run
```
python train_NetM.py
```
to train the adaptive image sampling network NetM.

To visualize the training process, run
```
tensorboard --logdir tensorBoardRuns
```

### Test NetE
```
python test_NetE_h5.py
```
The output reconstructed images are on `results/netE_results` directory.

### Test NetM
```
python test_NetM_h5.py
```
The output reconstructed images are on `results/netM_results` directory.

