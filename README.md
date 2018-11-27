# Adaptive Image Sampling using Deep Learning
A PyTorch implementation of the paper:
[Adaptive Image Sampling using Deep Learning and its Application on X-Ray Fluorescence Image Reconstruction](arxiv to be appear)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c pytorch
```
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

To visualize the training process log, run
```
tensorboard --logdir tensorBoardRuns
```

### Test Image
```
python test_image.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--model_name          super resolution model name [default value is epoch_3_100.pt]
```
The output high resolution images are on `results` directory.

### Test Video
```
python test_video.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--is_real_time        super resolution real time to show [default value is False]
--delay_time          super resolution delay time to show [default value is 1]
--model_name          super resolution model name [default value is epoch_3_100.pt]
```
The output high resolution videos are on `results` directory.

## Benchmarks
Adam optimizer were used with learning rate scheduling between epoch 30 and epoch 80.

**Upscale Factor = 2**

Epochs with batch size of 64 takes ~1 minute on a NVIDIA GeForce TITAN X GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/2_trainloss.png"/>
    </td>
    <td>
     <img src="images/2_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/2_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/2_valpsnr.png"/>
    </td>
  </tr>
</table>

> Image Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- Set5
<table>
  <tr>
    <td>
     <img src="images/2_LR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/2_HR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/2_SR_Set5_004.png"/>
    </td>
  </tr>
</table>

- Set14
<table>
  <tr>
    <td>
     <img src="images/2_LR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/2_HR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/2_SR_Set14_001.png"/>
    </td>
  </tr>
</table>

- BSD100
<table>
  <tr>
    <td>
     <img src="images/2_LR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/2_HR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/2_SR_BSD100_063.png"/>
    </td>
  </tr>
</table>

- Urban100
<table>
  <tr>
    <td>
     <img src="images/2_LR_Urban100_014.png"/>
    </td>
    <td>
     <img src="images/2_HR_Urban100_014.png"/>
    </td>
    <td>
     <img src="images/2_SR_Urban100_014.png"/>
    </td>
  </tr>
</table>

> Video Results

The right is low resolution video, the left is super resolution video(output of the ESPCN).
Click the image to watch the complete video.

[![Watch the video](images/video_SRF_2.png)](http://v.youku.com/v_show/id_XMzIwMDEyODU2MA==.html?spm=a2hzp.8244740.0.0)

**Upscale Factor = 3**

Epochs with batch size of 64 takes ~30 seconds on a NVIDIA GeForce TITAN X GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/3_trainloss.png"/>
    </td>
    <td>
     <img src="images/3_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/3_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/3_valpsnr.png"/>
    </td>
  </tr>
</table>

> Image Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- Set5
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set5_004.png"/>
    </td>
  </tr>
</table>

- Set14
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set14_001.png"/>
    </td>
  </tr>
</table>

- BSD100
<table>
  <tr>
    <td>
     <img src="images/3_LR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_HR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_SR_BSD100_063.png"/>
    </td>
  </tr>
</table>

> Video Results

The right is low resolution video, the left is super resolution video(output of the ESPCN). 
Click the image to watch the complete video.

[![Watch the video](images/video_SRF_3.png)](http://v.youku.com/v_show/id_XMzIwMDEzMjEyNA==.html?spm=a2hzp.8244740.0.0)

**Upscale Factor = 4**

Epochs with batch size of 64 takes ~20 seconds on a NVIDIA GeForce GTX 1070 GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/4_trainloss.png"/>
    </td>
    <td>
     <img src="images/4_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/4_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/4_valpsnr.png"/>
    </td>
  </tr>
</table>

> Image Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- Set5
<table>
  <tr>
    <td>
     <img src="images/4_LR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/4_HR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/4_SR_Set5_004.png"/>
    </td>
  </tr>
</table>

- Set14
<table>
  <tr>
    <td>
     <img src="images/4_LR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/4_HR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/4_SR_Set14_001.png"/>
    </td>
  </tr>
</table>

- BSD100
<table>
  <tr>
    <td>
     <img src="images/4_LR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/4_HR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/4_SR_BSD100_063.png"/>
    </td>
  </tr>
</table>

- Urban100
<table>
  <tr>
    <td>
     <img src="images/4_LR_Urban100_014.png"/>
    </td>
    <td>
     <img src="images/4_HR_Urban100_014.png"/>
    </td>
    <td>
     <img src="images/4_SR_Urban100_014.png"/>
    </td>
  </tr>
</table>

> Video Results

The right is low resolution video, the left is super resolution video(output of the ESPCN).
Click the image to watch the complete video.

[![Watch the video](images/video_SRF_4.png)](http://v.youku.com/v_show/id_XMzIwMDEzNDcxMg==.html?spm=a2hzp.8244740.0.0)

**Upscale Factor = 8**

Epochs with batch size of 64 takes ~15 seconds on a NVIDIA GeForce GTX 1070 GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/8_trainloss.png"/>
    </td>
    <td>
     <img src="images/8_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/8_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/8_valpsnr.png"/>
    </td>
  </tr>
</table>

> Image Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- SunHays80
<table>
  <tr>
    <td>
     <img src="images/8_LR_SunHays80_053.png"/>
    </td>
    <td>
     <img src="images/8_HR_SunHays80_053.png"/>
    </td>
    <td>
     <img src="images/8_SR_SunHays80_053.png"/>
    </td>
  </tr>
</table>

> Video Results

The left is low resolution video, the right is super resolution video(output of the ESPCN).
Click the image to watch the complete video.

[![Watch the video](images/video_SRF_8.png)](http://v.youku.com/v_show/id_XMzIwMDEzODMzNg==.html?spm=a2hzp.8244740.0.0)

The complete test image results could be downloaded from [here](https://pan.baidu.com/s/1eS5x5HC), and 
the complete test video results could be downloaded from [here](https://pan.baidu.com/s/1bZIvKU).
