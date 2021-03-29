'''
Code for the implementation of
"Spatially-Variant CNN-based Point Spread Function Estimation for Blind Deconvolution and Depth Estimation in Optical Microscopy"

Copyright (c) 2020 Idiap Research Institute, https://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of Spatially-Variant CNN-based Point Spread Function Estimation.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import numpy
from scipy import signal
import scipy
import math
import copy
import sys
import os
from skimage.transform import rotate
import bz2
import torch
import tifffile as tiff
from imageio import imread as io_imread
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

def multipage(filename, figs=None, dpi=200):
    '''
    Print all the plots in a PDF file
    '''
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def create_dir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def write_tiff_stack(file_name, array, compression=None, rgb=False):
    """ Script to export tif file to imageJ,

    usage: tiff.write_stack(out_file_name, array, compression=None)
    """
    if libtiff_OK:
        out_tiff = TIFF.open(file_name, mode='w')
        a = np.flipud(array)
        a = np.rollaxis(a, 3, 0)
        print('Lib import : ', libtiff_OK)
        for zInd in range(a.shape[3]):
            out_tiff.write_image(a[:, :, :, zInd], compression=compression, write_rgb=rgb)

        out_tiff.close()
    else:
        # for some reason, setting imagej=True gets an error from ImageJ with
        # 3D stacks (t,X,Y). Hence the comment hereunder
        imsave(file_name, array)  # , imagej=True)

    return None

def stretch_contrast(image, min_val=0.0, max_val=1.0):
    """ Rescales the greylevels in an image """
    curr_min = np.min(image)
    curr_max = np.max(image)
    image_ret = image - curr_min  # scale starts at zero
    ratio = (max_val-min_val) / (curr_max-curr_min)
    image_ret = image_ret * ratio + min_val

    return image_ret


def scale3d(v):
    '''
    Normalize a ND matrix with a maximum of 1 per pixel
    :param v:
    :return: normalized vector
    '''
    shape = v.shape
    norm = np.linalg.norm(v.flatten(), ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v.flatten() / norm
    return np.reshape(out * (1 / np.max(np.abs(out))), newshape=shape)


def pickle_save(filename, obj, compressed=True):
    """
    save object to file using pickle
    """

    try:
        if compressed:
            f = bz2.BZ2File(filename, 'wb')
        else:
            f = open(filename, 'wb')
    except IOError as details:
        sys.stderr.write('File {} cannot be written\n'.format(filename))
        sys.stderr.write(details)
        return

    pickle.dump(obj, f, protocol=2)
    f.close()


def plot_images(image_list, fig=None, title_str='', figure_title='',
                axes_titles=None, sub_plot_shape=None,
                is_blob_format=False, channel_swap=None, transpose_order=None,
                show_type=None):
    '''
    Plot list of n images
      image_list: list of images
      is_blob_format: Caffe stores images as ch x h x w
                    True - convert the images into h x w x ch format
      transpose_order     : If certain transpose order of channels is to be used
                    overrides is_blob_format
      show_type    : imshow or matshow (by default imshow)
    '''
    image_list = copy.deepcopy(image_list)
    if transpose_order is not None:
        for i, im in enumerate(image_list):
            image_list[i] = im.transpose(transpose_order)
    if transpose_order is None and is_blob_format:
        for i, im in enumerate(image_list):
            image_list[i] = im.transpose((1, 2, 0))
    if channel_swap is not None:
        for i, im in enumerate(image_list):
            image_list[i] = im[:, :, channel_swap]
    plt.ion()
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    plt.clf()
    if sub_plot_shape is None:
        N = np.ceil(np.sqrt(len(image_list)))
        sub_plot_shape = (N, N)
        # gs = gridspec.GridSpec(N, N)
    ax = []
    for i in range(len(image_list)):
        shp = sub_plot_shape + (i + 1,)
        aa = fig.add_subplot(*shp)
        aa.autoscale(False)
        if (len(figure_title) == len(image_list)):
            aa.set_title(figure_title[i])
        ax.append(aa)
        # ax.append(plt.subplot(gs[i]))

    if show_type is None:
        show_type = ['imshow'] * len(image_list)
    else:
        assert len(show_type) == len(image_list)

    for i, im in enumerate(image_list):
        ax[i].set_ylim(im.shape[0], 0)
        ax[i].set_xlim(0, im.shape[1])
        if show_type[i] == 'imshow':
            ax[i].imshow(im)
        elif show_type[i] == 'matshow':
            res = ax[i].matshow(im)
            plt.colorbar(res, ax=ax[i])
        ax[i].axis('off')
        if axes_titles is not None:
            ax[i].set_title(axes_titles[i])
    if len(figure_title) == 1:
        fig.suptitle(figure_title)
    #plt.show(block=True)

    return ax


def random_crop(img,N,onlygood=False,randx=None, randy=None):
    """ Randomly crop a portion of image of size NxN
        Onlygood -> selects for a part of the image with a good variance and mean
    """
    xmax = img.shape[0]-N
    ymax = img.shape[1]-N
    if xmax == 0 and ymax == 0:
        return img
    if xmax < 0 or ymax < 0:
        print('ERROR: the size of the input image is smaller than the crop size.')
        return img
    if onlygood == False:
        if randx is None and randy is None:
            randx = np.random.randint(0, xmax)
            randy = np.random.randint(0, ymax)
        return img[randx:randx+N , randy:randy+N]
    else:
        i = 0
        while i < 100:
            if randx is None and randy is None:

                if xmax == 0:
                    randx = 0
                else:
                    randx = np.random.randint(0, xmax)
                if ymax == 0:
                    randy = 0
                else:
                    randy = np.random.randint(0, ymax)
            im = img[randx:randx + N, randy:randy + N]

            sum_all_pixels = np.sum(im)
            variance = np.var(im)
            nb_pix = N ** 2
            ratio = sum_all_pixels / nb_pix
            i += 1
            if ratio > 0.12 and variance > 0.01:
                return im
                break
        print('Good ratio/variance not found.')
        return np.zeros((N,N))


def noisy(image, noise_type, param = 0.01):
    '''
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        'sp'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    '''
    noise_types = ['gauss', 'poisson', 'luminosity', 'rotation', 'axial_luminosity', 'poigauss']
    assert noise_type in noise_types, "ERROR: noise type {} does not exist".format(noise_type)

    if noise_type == 'gauss':
        var = param/150.0
        sigma = var ** 0.5
        mean = torch.zeros(image.size())
        sig =  sigma*torch.ones(image.size())
        gauss = torch.normal(mean, sig)
        noisy_image = image + gauss

    elif noise_type == "poisson":

        vals = len(np.unique(image.data.cpu().numpy()))*1.0/(420*param)
        vals = 2 ** np.ceil(np.log2(vals))
        if float(vals) == 0.0:
            return image
        noisy_image = torch.Tensor(np.random.poisson(image * vals) / float(vals))
    elif noise_type == "poigauss":
        return noisy(noisy(image, 'poisson', param), 'gauss', param)
    elif noise_type == "luminosity":
        noisy_image = image * (1.0-param)
    elif noise_type == "rotation":
        noisy_image = rotate(image, 180.00*param, resize=False, order=2)
        noisy_image = torch.Tensor(noisy_image)
    elif noise_type == "axial_luminosity":
        size_im = np.max(list(image.size()))
        mask = scale(gaussian_kernel(size_im, fwhmx = size_im/2, fwhmy = size_im/2, verbose=False))
        mask = torch.FloatTensor(mask)
        noisy_image = (1-param) * image + param * mask * image

    return noisy_image

def rand_int(a, b, size=1):
    '''
    Return random integers in the half-open interval [a, b).
    '''
    return np.floor((b - a) * np.random.random_sample(size) + a).astype(dtype=np.int16)


def normalize(v):
    '''
    Normalize a 2D matrix with a sum of 1
    :param v:
    :return: normalized vector
    '''
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def scale(v):
    '''
    Normalize a 2D matrix with a maximum of 1 per pixel
    :param v:
    :return: normalized vector
    '''
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    out = out * (1 / np.max(np.abs(out)))
    if np.all(np.isfinite(out)):
        return out
    else:
        print('Error, image is not finite (dividing by infinity on norm).')
        return np.zeros(v.shape)


def to_8_bit(v):
    '''
    Normalize a 32 bit float [0 1] image to [0 255] int 8 bit image.
    '''
    if isdtype(v, np.uint8):
        print('Warning: input already 8 bit.')
        return v

    return np.asarray(np.round(v * 255.0), dtype=np.uint8)


def to_16_bit(v):
    '''
    Normalize a 32 bit float [0 1] image to [0 65535] int 16 bit image.
    '''
    if isdtype(v, np.uint16):
        print('Warning: input already 16 bit.')
        return v

    return np.asarray(np.round(v * 65536.0), dtype=np.uint16)


def to_32_bit(v):
    '''
    Normalize [0 255] int 8 bit image to a 32 bit float [0 1] image.
    '''
    if isdtype(v, np.float32):
        print('Warning: input already 32 bit.')
        return v

    return np.asarray(v, dtype=np.float32) / 255.0


def unpad(img, npad):
    '''
    Revert the np.pad command
    '''
    return img[npad:-npad, npad:-npad]


def to_radial(x, y):
    return x ** 2 + y ** 2


def to_radian(x):
    return float(x) * np.pi / 180.


def isdtype(a, dt=np.float64):
    '''
    Test for type
    '''
    try:
        return a.dtype.num == np.dtype(dt).num
    except AttributeError:
        return False


def center_crop(img, percentage):
    '''
    Extract center crop
    :param img: input image
    :param percentage: percentage of area to keep
    '''
    assert (img.shape[0] == img.shape[1])
    a = img.shape[0]
    offset = int(round(0.5 * a * (1 - np.sqrt(percentage / 100.0))))
    return img[offset:-offset, offset:-offset]


def center_crop_pixel(img, size):
    '''
    Extract center crop
    :param img: input image
    :param pixel size of the patch
    '''
    assert (img.shape[0] == img.shape[1])
    if img.shape[0] == size:
        return img
    assert (img.shape[0] > size)
    margin_size = img.shape[0] - size
    if margin_size % 2 == 0:
        return unpad(img, int(margin_size / 2))
    else:
        npad = int(math.floor(margin_size / 2))
        return img[npad:-npad, npad + 1:-npad + 1]


def gaussian_kernel(size, fwhmx = 3, fwhmy = 3, center=None, verbose=True):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        if size % 2 == 0 and verbose:
            print("WARNING gaussian_kernel : you have chosen a even kernel size and therefore the kernel is not centered.")
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    if fwhmx == 0 and fwhmx == 0:
        ret = np.zeros((size, size))
        ret[x0,y0] = 1.0
        return ret
    else:
        return normalize(np.exp(-4 * np.log(2) * ( ((x - x0) ** 2) / fwhmx**2 + ((y - y0) ** 2) / fwhmy**2 )))


def pickle_load(filename, compressed=True):
    """
    Load from filename using pickle
    """

    try:
        if compressed:
            f = bz2.BZ2File(filename, 'rb')
        else:
            f = open(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File {} cannot be read\n'.format(filename))
        sys.stderr.write(details)
        return

    obj = pickle.load(f)
    f.close()
    return obj


def convolve(input, psf, padding = 'constant'):
    '''
    Convolve an image with a psf using FFT
    :param padding: replicate, reflect, constant
    :return: output image
    '''
    psf = normalize(psf)
    npad = np.max(psf.shape)

    if len(input.shape) != len(psf.shape):
        #print("Warning, input has shape : {}, psf has shape : {}".format(input.shape, psf.shape))
        input = input[:,:,0]
        #print("New input shape : {}".format(input.shape))

    input = np.pad(input, pad_width=npad, mode=padding)

    try:
        out = scipy.signal.fftconvolve(input, psf, mode='same')
    except:
        print("Exception: FFT cannot be made on image !")
        out = np.zeros(input.shape)

    out = unpad(out, npad)
    return out


def rand_float(a, b, size=1):
    '''
    Return random floats in the half-open interval [a, b).
    '''
    return (b - a) * numpy.random.random_sample(size) + a


def rand_int(a, b, size=1):
    '''
    Return random integers in the half-open interval [a, b).
    '''
    return numpy.floor((b - a) * numpy.random.random_sample(size) + a).astype(dtype=numpy.int16)


def get_wavefront(x,y,params):
    x = 2.*x/params.size
    y = 2.*y/params.size
    r2 = to_radial(x, y)

    aberration = params.sph * r2**2 + params.focus * r2 + params.ast * (x*np.cos(params.ast_angle) + y*np.sin(params.ast_angle))**2 + \
                 params.coma * ( (x*r2)*np.cos(params.coma_angle) + (y*r2)*np.sin(params.coma_angle)) + \
                 params.tilt*(x*np.cos(params.tilt_angle) + y*np.sin(params.tilt_angle))
    wavefront = np.exp(2*1j*np.pi*aberration)
    return wavefront


class Params:
    def __init__(self):
        self.tilt = 0.
        self.tilt_angle = to_radian(0.)
        self.focus = 0.
        self.coma = 0.
        self.coma_angle = to_radian(0.)
        self.ast = 0.
        self.ast_angle = to_radian(0.)
        self.sph = 0.
        self.size = 127. # px
        self.wavelength=570. # nm
        self.tubelength=200. # mm
        self.na = 0.8
        self.n = 1.1 #refraction
        self.magnification = 20.
        self.pixelsize=10 # um


def get_psf(params, centered = True):
    datapoints = params.size
    padding = int(np.ceil(datapoints/2))
    totalpoints = datapoints + 2*padding
    center_point = int(np.floor(totalpoints/2))

    wavelength = params.wavelength * float(1e-9) #wavelength in m
    pupil_diameter = 2.0 * params.tubelength * params.na / (params.magnification * params.n)
    D = pupil_diameter*1e-3 # diameter in m
    d = 1.0*1e-2 # distance btw pupil plane and object
    PRw = D / (2 * wavelength * d) # unit = 1/m
    NT = params.size
    x = np.linspace(-NT/2, NT/2, datapoints)
    y = np.linspace(-NT/2, NT/2, datapoints)
    xx, yy = np.meshgrid(x, y)
    sums = np.power(xx,2) + np.power(yy,2)
    wavefront = get_wavefront(xx, yy, params)
    pixel_limit = PRw*params.size*params.pixelsize*1e-6

    wavefront[sums > pixel_limit] = 0
    wavefront_padded = np.pad(wavefront, ((padding,padding),(padding,padding)), mode='constant',constant_values=(0))

    psf = np.power(np.abs(np.fft.fft2(wavefront_padded, norm='ortho')),2)
    psf = np.roll(psf, center_point, axis = (0,1))

    normalisation = np.power(np.sum(np.abs(wavefront)) / float(totalpoints),2)
    psf = unpad(psf, padding) / normalisation
    psf = scale(np.fliplr(psf)).astype(np.float32)

    return psf, wavefront, pupil_diameter
