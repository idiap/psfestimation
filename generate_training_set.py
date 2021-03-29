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

from toolbox import *
from pandas.io.parsers import read_csv
import glob
import random
import numpy as np
from skimage import io
import argparse
import h5py
import imageio


def get_parser():
    parser = argparse.ArgumentParser(description='Create dataset for regression of PSFS')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=128)
    parser.add_argument('--psf_size', dest='psf_size', type=int, default=127)
    parser.add_argument('--name', dest='name', type=str, default='images')
    parser.add_argument('--synthetic', dest='synthetic', type=int, default=0)
    parser.add_argument('--natural', dest='natural', type=int, default=1)
    parser.add_argument('--points', dest='points', type=int, default=0)
    parser.add_argument('--black', dest='black', type=int, default=5)
    parser.add_argument('--noise', dest='noise', type=int, default=1)
    parser.add_argument('--type', dest='type', type=str, default='1dgaussian')
    return parser


file_list = glob.glob("../input_images/*")
random.shuffle(file_list)
num = int(round(len(file_list)*0.9))
training_file_list, test_file_list = file_list[:num], file_list[num:]


def handle_list(output_dir,is_train=True):

    if model_type == '1dgaussian':
        header = 'isfake,fwmh\n'
    elif model_type == '2dgaussian':
        header = 'isfake,fwmhx,fwmhy\n'
    elif model_type == '1dzernike':
        header = 'isfake,focus\n'
    elif model_type == '2dzernike':
        header = 'isfake,focus,ast,ast_angle\n'
    else:
        print('Undefined model type')
        exit()
    output_dir = input_dir+output_dir
    create_dir(output_dir)
    list_file = open("{}/parameters.txt".format(output_dir), 'w')
    list_file.write(header)

    i = 0

    def handle_file(original_img, filenb, i, is_synth=False, is_black=False):

        for a in range(30):
            img = original_img.copy()
            if model_type == '1dgaussian':
                img, params = do_convolution_gaussian(img, dimensions=1)
                str_params_valid = '{},{}\n'.format(0, params['fwmhx'])
                str_params_invalid = '{},{}\n'.format(1, 1000)
            elif model_type == '2dgaussian':
                img, params = do_convolution_gaussian(img, dimensions=2)
                str_params_valid = '{},{},{}\n'.format(0, params['fwmhx'], params['fwmhy'])
                str_params_invalid = '{},{},{}\n'.format(1, 1000, 1000)
            elif model_type == '1dzernike':
                img, params = do_convolution_zernike(img, dimensions=1)
                str_params_valid = '{},{}\n'.format(0, params.focus)
                str_params_invalid = '{},{}\n'.format(1, 1000)
            elif model_type == '2dzernike':
                img, params = do_convolution_zernike(img, dimensions=2)
                str_params_valid = '{},{},{},{}\n'.format(0, params.focus, params.ast, params.ast_angle)
                str_params_invalid = '{},{},{},{}\n'.format(1, 1000,1000,1000,1000)
            else:
                print('Undefined model type')
                exit()

            img = img.astype(np.float)

            if (np.min(img) < 0.0):
                img -= np.min(img)

            size = args.patch_size
            step = 48
            x = size
            y = size

            if img.max() > 0:
                img -= img.mean()
                img /= (img.std())
                img += 1
            if img.max() > 1:
                img /= img.max()

            while x <= img.shape[0]:
                while y <= img.shape[1]:
                    im = img[x-size:x,y-size:y]

                    sum_all_pixels = np.sum(im)
                    nb_pix = size**2
                    variance = np.var(im)/nb_pix/255.0

                    ratio = sum_all_pixels/nb_pix/255.0

                    if args.noise > 0.0 and is_train:
                        im = noisy(im, 'gauss', rand_float(0.00001, 0.002))
                        im = noisy(im, 'poisson', rand_float(0.00001, 0.001))
                    if is_train:
                        coeff = rand_float(0.4, 1.0, 1)
                        im *= coeff
                    if im.min() < 0:
                        im +=im.min()
                    if im.max() > 1:
                        im /= im.max()
                    im = to_8_bit(im)

                    print ('{} Ratio : {} | variance : {}'.format(i, ratio,variance))
                    filename = "{}/{:05d}/{:09d}.png".format(output_dir,filenb, i)
                    saved = False
                    if ((ratio > 0.0003 and variance > 1e-09) or is_synth) and not is_black:
                        list_file.write(str_params_valid)
                        print('Good min {} max {}'.format(im.min(), im.max()))
                        saved = True

                    else:
                        if ratio < 0.00015 or variance < 4e-11 or is_black:
                            list_file.write(str_params_invalid)
                            print('As fake')
                            saved = True
                        else:
                            pass
                            print('Rejected')
                    if saved:
                        i += 1
                        imageio.imsave(filename, im)
                    y += step
                y = size
                x += step
        return i

    filenb = 0

    if args.natural > 0:
        if is_train:
            ff = training_file_list
        else:
            ff = test_file_list
        for file in ff:
            print('Loading file {}'.format(file))
            original_img = io.imread(file)
            original_img = scale(original_img)
            create_dir(output_dir + '/{:05d}'.format(filenb))
            i = handle_file(original_img, filenb, i)
            filenb += 1

    if args.synthetic > 0:

        if not is_train:
            numsynth = int(args.synthetic//10+1)
        else:
            numsynth = args.synthetic

        for a in range(0,numsynth):
            synthetic_image = random_generate((600,600), number_coef=35, size_coeff=0.25, size_variance_coeff = 1.5, noise='poisson', noise_coeff=1.0)
            create_dir(output_dir + '/{:05d}'.format(filenb))
            i = handle_file(synthetic_image, filenb, i, True)
            filenb +=1

    if args.points > 0:

        if not is_train:
            numsynth = int(args.points//10+1)
        else:
            numsynth = args.points

        for a in range(0,numsynth):
            synthetic_image = random_generate((600, 600), number_coef=100, size_coeff=0, size_variance_coeff=0.7,
                                          noise='poisson', noise_coeff=1.0)
            create_dir(output_dir + '/{:05d}'.format(filenb))
            i = handle_file(synthetic_image, filenb, i, True, is_black=False)
            filenb +=1

    if args.black > 0:
        if not is_train:
            numsynth = int(args.black//10+1)
        else:
            numsynth = args.black

        for a in range(0,numsynth):
            synthetic_image = np.zeros((800,800))
            create_dir(output_dir + '/{:05d}'.format(filenb))
            i = handle_file(synthetic_image, filenb, i, is_black=True)
            filenb +=1

    list_file.close()


def do_convolution_gaussian(img, dimensions=1):
    small_rand = rand_float(0, 2, 1)
    large_rand = rand_float(5, 20, 1)
    iso_rand = rand_float(0,20,1)
    choice = rand_int(0,3)
    if dimensions == 1:
        choice = 2

    if choice == 0:
        params = {'fwmhx': small_rand[0], 'fwmhy': large_rand[0]}
    elif choice == 1:
        params = {'fwmhx': large_rand[0], 'fwmhy': small_rand[0]}
    elif choice == 2:
        params = {'fwmhx': iso_rand[0], 'fwmhy': iso_rand[0]}
    else:
        return

    psf = gaussian_kernel(args.psf_size, params['fwmhx'], params['fwmhy'])
    return [convolve(img, psf, padding='reflect'), params]



def do_convolution_zernike(img, dimensions=1):
    small_rand = rand_float(0, 3, 1)

    if(random.randint(0,100) > 50 or (dimensions == 2 and small_rand[0] > 0.5)):
        large_rand = rand_float(0, 1.0)
    else:
        large_rand = rand_float(0, 4.0)
    rand_angle = random.choice([0,math.pi/2.0])
    params = Params()
    params.magnification = 20
    params.n = 1.33
    params.na = 0.45
    params.wavelength = 500
    params.pixelsize = 45
    params.tubelength = 200
    params.size = args.psf_size
    params.focus = large_rand[0]
    if dimensions == 1:
        params.ast = 0.0
    else:
        params.ast = small_rand[0]
        params.ast_angle = rand_angle
    params.sph = 0.0
    psf, wavefront, pupil_diameter = get_psf(params)
    return [convolve(img, psf), params]


def create_hdf5(output_dir):
    output_dir = input_dir+output_dir
    all_files_list = sorted(glob.glob(output_dir+"/*/*.png"),key=lambda name: int(name[-13:-4]))
    num_files  = len(all_files_list)
    print('{} files found'.format(num_files))
    try:
        os.remove(output_dir+'data.h5')
    except OSError:
        pass

    with h5py.File(output_dir+'data.h5', 'w') as f:

        _file_csv = read_csv(os.path.expanduser(output_dir + "parameters.txt"))
        _file = _file_csv.values.astype(np.float)
        _header = _file_csv.head(0).columns.base

        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        data = f.create_dataset('data', (num_files, ), dtype=dt)
        labels = f.create_dataset('labels', (num_files, len(_header)))

        i = 0
        a = 0
        num_chunk = 1
        max_files_chunk = 1000
        image_list = []
        labels_list = []
        for filename in all_files_list:

            if a == max_files_chunk or i == num_files-1:
                f['data'][i-a:i] = image_list
                f['labels'][i-a:i]  = labels_list

                image_list = []
                labels_list = []
                a = 0
                print("Chunk {}/{} written...".format(num_chunk, num_files//max_files_chunk +1))
                num_chunk += 1

            image = open(filename, 'rb').read()

            index = int(filename[-13:-4])
            labels_list.append(_file[index])
            image_list.append(np.fromstring(image, dtype='uint8'))

            a += 1
            i += 1
        print("Finished!")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    model_type = args.type
    input_dir = 'data/'
    train_dir = 'psf_{}_n_{}_s_{}_p_{}_b_{}_{}_train/'.format(args.patch_size, args.natural, args.synthetic, args.points, args.black, model_type)
    print('Directory:{}'.format(train_dir))
    handle_list(train_dir, True)
    create_hdf5(train_dir)
    test_dir = 'psf_{}_n_{}_s_{}_p_{}_b_{}_{}_test/'.format(args.patch_size, args.natural, args.synthetic, args.points, args.black,model_type)
    print('Directory:{}'.format(test_dir))
    handle_list(test_dir, False)
    create_hdf5(test_dir)
