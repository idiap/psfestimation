# PSF Estimation
Code for the PyTorch implementation of "Spatially-Variant CNN-based Point Spread Function Estimation for Blind Deconvolution and Depth Estimation in Optical Microscopy", IEEE Transactions on Image Processing, 2020.

https://ieeexplore.ieee.org/document/9068472

## Abstract
Optical microscopy is an essential tool in biology and medicine. Imaging thin, yet non-flat objects in a single shot (without relying on more sophisticated sectioning setups) remains challenging as the shallow depth of field that comes with highresolution microscopes leads to unsharp image regions and makes depth localization and quantitative image interpretation difficult. Here, we present a method that improves the resolution of light microscopy images of such objects by locally estimating image distortion while jointly estimating object distance to the focal plane. Specifically, we estimate the parameters of a spatiallyvariant Point Spread Function (PSF) model using a Convolutional Neural Network (CNN), which does not require instrument- or object-specific calibration. Our method recovers PSF parameters from the image itself with up to a squared Pearson correlation coefficient of 0.99 in ideal conditions, while remaining robust to object rotation, illumination variations, or photon noise. When the recovered PSFs are used with a spatially-variant and regularized Richardson-Lucy (RL) deconvolution algorithm, we observed up to 2.1 dB better Signal-to-Noise Ratio (SNR) compared to other Blind Deconvolution (BD) techniques. Following microscope-specific calibration, we further demonstrate that the recovered PSF model parameters permit estimating surface depth with a precision of 2 micrometers and over an extended range when using engineered PSFs. Our method opens up multiple possibilities for enhancing images of non-flat objects with minimal need for a priori knowledge about the optical setup.

## Requirements
The following python libraries are required. We advise the use of the conda package manager.
> numpy
> scikit-image
> pytorch
> matplotlib
> PyQt5
> pandas
> scikit-learn

For example, you can install all the requirements by using
> conda install --file requirements.txt

## Generating training dataset
Launch the file `generate_training_set.py` with the according parameters

## Training
Launch `train.py` and modify the parameters to match the training set folder.

## Deconvolution
The code for deconvolution is in the separate directory `https://github.com/idiap/semiblindpsfdeconv`

## Generating figures and tables
The benchmark table is in file `benchmark_models.py`; noise resistance figure in `figure_noise_resistance.py`, and the depth-from-focus figure in `figure_depth.py`

## Citation
For any use of the code or parts of the code, please cite:

@article{shajkofci_spatially-variant_2020,
  ids = {shajkofci\_spatially-variant\_2020},
  title = {Spatially-{{Variant CNN}}-{{Based Point Spread Function Estimation}} for {{Blind Deconvolution}} and {{Depth Estimation}} in {{Optical Microscopy}}},
  author = {Shajkofci, Adrian and Liebling, Michael},
  date = {2020},
  journaltitle = {IEEE Transactions on Image Processing},
  volume = {29},
  pages = {5848--5861},
  issn = {1941-0042},
  doi = {10.1109/TIP.2020.2986880},
  eventtitle = {{{IEEE Transactions}} on {{Image Processing}}},
  keywords = {blind deconvolution,Calibration,convolutional neural networks,Deconvolution,depth from focus,Estimation,Microscopy,Optical diffraction,Optical imaging,Optical microscopy,point spread function estimation}
}


## Licence
This is free software: you can redistribute it and/or modify it under the terms of the BSD-3-Clause licence.
