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


import pandas
import pprint
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

import datetime
from toolbox import random_open_crop, random_crop
from toolbox import to_32_bit, scale, to_16_bit
from toolbox import center_crop_pixel

from evaluation_accuracy import *
from train import *

torch.random.manual_seed(8)
pp = pprint.PrettyPrinter()

directory="data/models/"

file_list = glob.glob(directory+"*")

all_params = []
for file in file_list:
    if 'texture'  in file:
        continue
    parsed = file.replace(directory, '')
    has_nat = 0
    has_nat_mult = 0
    if '_natural_' in parsed:
        has_nat = 1
        has_nat_mult = 1
    elif '_nonoise_' in parsed:
        has_nat = 1
        has_nat_mult = 2
    parsed = parsed.replace('.pt', '')
    parsed = parsed.split("_")
    params = {}
    params['patch_size'] = int(parsed[1])
    params['train_natural'] = int(parsed[3])+2*has_nat_mult
    params['train_synthetic'] = parsed[5]
    params['train_points'] = parsed[7]
    params['train_black'] = parsed[9]
    params['dataset_trained'] = parsed[has_nat+10]
    params['run'] = int(parsed[has_nat+11])
    params['model'] = parsed[has_nat+12]
    found = list(filter(lambda file: file['train_natural'] == params['train_natural'] and file['train_synthetic'] == params['train_synthetic'] and file['train_points'] == params['train_points']
            and file['train_black'] == params['train_black'] and file['dataset_trained'] == params['dataset_trained'] and file['run'] == params['run'] and file['model'] == params['model'], all_params))
    # if len(found) > 0:
    #     found[0]['train_err'].append((int(parsed[has_nat+13][2:]), float(parsed[has_nat+14][8:])))
    #     found[0]['test_err'].append((int(parsed[has_nat+13][2:]), float(parsed[has_nat+15][7:])))
    #     found[0]['files'].append((int(parsed[has_nat+13][2:]),file))
    #     found[0]['train_err'] = sorted(found[0]['train_err'], key=lambda tup: tup[0])
    #     found[0]['test_err'] = sorted(found[0]['test_err'], key=lambda tup: tup[0])
    #     found[0]['files'] = sorted(found[0]['files'], key=lambda tup: tup[0])
    #
    # else:
    params['train_err'] = []
    params['test_err'] = []
    params['train_err'].append((int(parsed[has_nat+13][2:]), float(parsed[has_nat+14][8:])))
    params['test_err'].append((int(parsed[has_nat+13][2:]), float(parsed[has_nat+15][7:])))
    params['files'] = []
    params['files'].append((int(parsed[has_nat+13][2:]),file))
    all_params.append(params)

for p in all_params:
    train_err = [err[1] for err in p['train_err']]
    test_err = [err[1] for err in p['test_err']]
    files = [err[1] for err in p['files']]
    p.update({'train_err':train_err})
    p.update({'test_err':test_err})
    p.update({'files': files})
    p.update({'best_epoch_train': np.argmin(train_err)})
    p.update({'best_epoch_test': np.argmin(test_err)})

print('Found {} different modalities'.format(all_params.__sizeof__()))

def test(params):
    global log

    file = params['files'][params['best_epoch_train']]
    patch_size = params['patch_size']
    model_type = params['dataset_trained']
    synthetic = params['test_synthetic']
    natural = params['test_natural']
    points = params['test_points']
    black = params['test_black']

    if 'noise' in params:
        noise = params['noise']
        noise_type = params['noise_type']
    else:
        noise = 0.0
        noise_type = None
    if natural == 3:
        isnat = '_natural'
        natural = 1
    elif natural == 5: # nonoise
        isnat = ''
        natural = 1
    else:
        isnat=''

    run_nb = 'test'
    run_name = '{}_n_{}_s_{}_p_{}_b_{}{}_{}_{}/'.format(patch_size, natural, synthetic, points, black, isnat,model_type, run_nb)
    folder_prefix = "/idiap/temp/ashajkofci/"

    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name),
        handlers=[
            logging.StreamHandler()
        ])

    log = logging.getLogger('')
    log.setLevel(logging.INFO)

    def add_running_mean(_node):
        for child in _node.children():
            if type(child) == nn.BatchNorm2d:
                if child.running_mean is None:
                    del child._parameters['running_mean']
                    del child._parameters['running_var']
                    del child._parameters['num_batches_tracked']
                    child.register_buffer('running_mean', torch.zeros(child.num_features).cuda())
                    child.register_buffer('running_var', torch.ones(child.num_features).cuda())
                    child.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).cuda())
                    child.track_running_stats = True
            elif isinstance(child, nn.Module):
                add_running_mean(child)
        return


    def stop_running_mean(_node):
        for child in _node.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
            elif isinstance(child, nn.Module):
                stop_running_mean(child)
        return

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name+file.split('/')[-1]),
        handlers=[
            logging.FileHandler("output_log_{}.log".format(run_nb)),
            logging.StreamHandler()
        ])

    log.info("Load test data for {}....".format(run_name))
    log.info("Loading file {}".format(file))
    model = torch.load('{}'.format(file))
    model.eval()

    test_loader_, test_header_ = load_crops(folder_prefix = folder_prefix, test=True, patch_size=patch_size, synthetic=synthetic, natural=natural, points=points, black=black, model_type=model_type, batch_size=84, noise=noise, isnew=False, noise_type=noise_type, suffix=isnat)
    results = eval_regression_accuracy(model, test_loader_, test_header_, max_iter=10000)

    return results

def test_everything():
    df = None
    for p in all_params:

        if int(p['run']) < 600 or int(p['run']) > 700:
            continue

        tested_parameters = []
        all_train_set = list(filter(lambda file: file['model'] == p['model'] and file['dataset_trained'] == p['dataset_trained'] and file['patch_size'] == p['patch_size'] , all_params))
        for item in all_train_set:
            p2 = p.copy()

            p2['test_synthetic'] = item['train_synthetic']
            p2['test_natural'] = item['train_natural']
            p2['test_points'] = item['train_points']
            p2['test_black'] = item['train_black']

            p2['test_synthetic'] = p2['train_synthetic']
            p2['test_natural'] = p2['train_natural']
            p2['test_points'] = p2['train_points']
            p2['test_black'] = p2['train_black']

            param_tuple = (item['train_synthetic'],item['train_natural'],item['train_points'],item['train_black'])
            if param_tuple in tested_parameters:
                continue
            tested_parameters.append(param_tuple)
            results = test(p2)
            p2.update(results)
            for key, val in p.items():
                if isinstance(val, list):
                    p2[key] = ' '.join(str(e) for e in val)
            if df is None:
                df = pandas.DataFrame(p2, index=[0])
            else:
                df = pandas.concat([df, pandas.DataFrame(p2, index=[0])], axis=0, sort=False)
            df.to_csv('results_nonoise_test.csv', float_format='%.10f')
            break


def test_noise():
    noise_levels = np.linspace(0.0,1.0,50)
    df = None
    runs= [263,164,102,146,605,609]

    runsf = [str(i) for i in runs]

    noise_type = 'axial_luminosity'

    filename = 'results_noise_{}_{}_{}_{}_{}.csv'.format(noise_type, np.min(noise_levels) , np.max(noise_levels), noise_levels.shape[0] , '-'.join(runsf))
    print("Saving in {}".format(filename))

    all_train_set = list(filter(
        lambda file: file['run'] in runs, all_params))
    for p in all_train_set:
        for noise in noise_levels:
            log.info("Noise level: {}".format(noise))
            log.info("Noise type: {}".format(noise_type))
            p2 = p.copy()
            p2['noise'] = noise
            p2['noise_type'] = noise_type
            p2['test_synthetic'] = p['train_synthetic']
            p2['test_natural'] = p['train_natural']
            p2['test_points'] = p['train_points']
            p2['test_black'] = p['train_black']
            results = test(p2)
            p2.update(results)
            for key, val in p.items():
                if isinstance(val, list):
                    p2[key] = ' '.join(str(e) for e in val)
            if df is None:
                df = pandas.DataFrame(p2, index=[0])
            else:
                df = pandas.concat([df, pandas.DataFrame(p2, index=[0])], axis=0, sort=False)
            df.to_csv(filename, float_format='%.10f')


def print_noisy():
    images = random_open_crop('/media/adrian/ext4data/data/images_texture2/', None, num_images = 3, as_grey=True)
    ranges_large = [0.1, 0.5, 0.9]
    i = 0
    def to_long(img):
        if img.min() < 0:
            img -= img.min()
        if img.max() > 1:
            img /= img.max()
        img *= 65535
        
        return img.astype(np.uint16)
    
    for img in images:

        img = img.astype(np.float)
        img = scale(img)
        if (np.min(img) < 0.0):
            img -= np.min(img)
        if img.max() > 0:
            img -= img.mean()
            img /= (img.std())
            img += 1
        if img.max() > 1:
            img /= img.max()
        if img.min() < 0:
            img -= img.min()
        if img.max() > 1:
            img /= img.max()
        image = Tensor(img)

        for r in ranges_large:
            io.imsave('noisy_gaussian_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(image, 'gauss', r).data.cpu().numpy(), 128)))
            io.imsave('noisy_poisson_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(image, 'poisson', r).data.cpu().numpy(), 128)))
            io.imsave('noisy_gaussianpoisson_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(noisy(image, 'poisson', r), 'gauss', r).data.cpu().numpy(), 128)))
            io.imsave('noisy_rotation_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(image, 'rotation', r).data.cpu().numpy(), 128)))
            io.imsave('noisy_luminosity_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(image, 'luminosity', r).data.cpu().numpy(), 128)))
            io.imsave('noisy_axialluminosity_{}_{}.png'.format(i, r), to_long(center_crop_pixel(noisy(image, 'axial_luminosity', r).data.cpu().numpy(), 128)))
        i += 1


def test_plane_graph(feat13_arr, feat23_arr, num_cases, step, figname='', degree=6, verbose=True):

    degrees_inclination = degree
    radians_inclination = degrees_inclination * math.pi / 180.
    resolution = 0.32
    size = feat13_arr[0].shape[0]
    max_view = resolution*size #micrometers 0.32 micron per pixel * 2048 pixel
    max_depth = max_view * math.tan(radians_inclination) *2.0
    print('Max depth : {} um'.format(max_depth))

    if verbose:
        plt.figure(dpi=600, figsize=(4,3))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    for idx, feat13 in enumerate(feat13_arr):

        feat13 = feat13[:,1::step]
        y_13 = feat13.mean(axis=0)
        x = np.linspace(0,feat13.shape[1],feat13.shape[1])
        var = feat13.var(axis=0)
        alpha = max_depth / (y_13.max()-y_13.min())
        y = y_13*alpha
        x = x*max_view/num_cases[idx]

        fit = np.polyfit(x,y,1)
        fit_fn = np.poly1d(fit)
        fitted_x = fit_fn(x)
        rsquare = r2_score(y, fitted_x)
        error1 = ((y - fitted_x).__abs__())
        #if verbose:
        #    plt.plot(x,y, '.', label=r'$z(\mathbf{s})$')
        #    plt.plot(x, fitted_x, '--k', label='gt')
        #    plt.errorbar(x,y,yerr=var*max_view)
        #    print('Y range :{}'.format((y.max()-y.min())))
        #   print('Image {} | Feature 1+3 : R2= {} err= {}'.format(idx, rsquare, error1.mean()))

        feat23 = feat23_arr[idx]
        feat23 = feat23[:,1::step]
        y_23 = feat23.mean(axis=0)
        x = np.linspace(0,feat23.shape[1],feat23.shape[1])
        var = feat23.var(axis=0)
        alpha = max_depth / (y_23.max()-y_23.min())
        y2 = y_23*alpha
        x = x*max_view/num_cases[idx]

        fit = np.polyfit(x,y2,1)
        fit_fn = np.poly1d(fit)
        fitted_x = fit_fn(x)
        rsquare2 = r2_score(y2, fitted_x)
        error2 = ((y2 - fitted_x).__abs__())
        if verbose:
            print('Y range :{}'.format((y2.max()-y2.min())))

            plt.plot(x,y2, '.', label=r'$z(\mathbf{s})$')
            plt.plot(x, fitted_x, '--k', label='gt')
            plt.errorbar(x,y2,yerr=var*max_view)
            print('Image {} | Feature 2+3 : R2= {} err= {}'.format(idx, rsquare2, error2.mean()))

    if verbose:
        #plt.title(r'$R^2=$' + '{:0.3f} / error = {:0.5f}'.format(rsquare, error))

        plt.xlabel(r'x position $(\mu m)$')
        plt.ylabel(r'z position $(\mu m)$')
        data = np.asarray([y2.max(), y2.min()])
        plt.ylim(data.min()-5, data.max()+5)
        #plt.title(figname)
        plt.legend()
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.22)
        plt.savefig('line_plot.png')

        # plt.figure(dpi=300, figsize=(5,3))
        # plt.plot(x, error1, '.')
        # plt.plot(x, error2, '.')
        # plt.xlabel('Y direction in the input image [microns]')
        # plt.ylabel('Depth [microns]')
        # plt.legend(['error 1+3', 'error 2+3'])
        # plt.title(figname)
        plt.show()

    print('R2 1+3 = {} R2 2+3 = {}'.format(rsquare, rsquare2))
    print('Error 1+3 = {} error 2+3 = {}'.format(error1.mean(), error2.mean()))
    return rsquare, rsquare2, error1.mean(), error2.mean(), error1.std(), error2.std()


def test_plane_stats(model_file=None, step=128):
    all_data = []
    degrees = [3,6,10]
    folder_tpl = '/home/adrian/git/adrian-wip-git/gaussiandeconv/img/astigmatism_avril2019/proche/{}/*/*.tif'

    if model_file is None:
        folder_model_tpl = '/media/adrian/OMENDATA/data/trained_models_new/*2d*.pt'
    else:
        folder_model_tpl = model_file

    files = glob.glob(folder_model_tpl)
    for model_file in files:
        print("Load model {}".format(model_file))
        for degree in degrees:
            folder = folder_tpl.format(degree)
            print('Search for {}'.format(folder))
            img_list = glob.glob(folder)
            print('Found {} files'.format(len(img_list)))
            for im_file in img_list:
                feat13_arr = []
                feat23_arr = []
                num_cases_arr = []
                feat13, feat23, num_cases = test_moving_window(model_file, im_file, step=step, verbose=False)
                feat13_arr.append(feat13)
                feat23_arr.append(feat23)
                num_cases_arr.append(num_cases)

                rsquare, rsquare2, error1, error2, e1std, e2std = test_plane_graph(feat13_arr, feat23_arr, num_cases_arr, step, '{} degrees'.format(degree), degree=degree, verbose=True)
            all_data.append({'model_file':model_file, 'degree':degree, 'rsquare1':rsquare, 'rsquare2':rsquare2, 'error1':error1, 'error2':error2, 'error1_std':e1std, 'error2_std':e2std})
        df = pandas.DataFrame(all_data)
        df.to_csv('plane_stats_results_{}.csv'.format(datetime.datetime.now()),float_format='%.5f')


model_file = None
image_file = []

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.folderLayout = QWidget();

        self.pathRoot = QDir.rootPath()

        self.dirmodel = QFileSystemModel(self)
        self.dirmodel.setRootPath(QDir.currentPath())

        self.indexRoot = self.dirmodel.index(self.dirmodel.rootPath())

        self.folder_view = QTreeView();
        self.folder_view.setDragEnabled(True)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.setRootIndex(self.indexRoot)

        self.selectionModel = self.folder_view.selectionModel()

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.folder_view)

        self.folderLayout.setLayout(self.left_layout)

        splitter_filebrowser = QSplitter(Qt.Horizontal)
        splitter_filebrowser.addWidget(self.folderLayout)
        splitter_filebrowser.addWidget(Figure_Canvas(self))
        splitter_filebrowser.setStretchFactor(1, 1)

        self.textbox = QLineEdit(self)
        self.textbox.resize(50, 40)
        self.textbox.setText('128')
        button = QPushButton('Load', self)
        button.clicked.connect(self.on_click)

        button2 = QPushButton('Line graph', self)
        button2.clicked.connect(self.on_click2)

        vbox = QVBoxLayout()
        vbox.addWidget(self.textbox)
        vbox.addWidget(button)
        vbox.addWidget(button2)

        hbox = QHBoxLayout(self)
        hbox.addWidget(splitter_filebrowser)
        hbox.addLayout(vbox)

        self.centralWidget().setLayout(hbox)

        self.setWindowTitle('PSF detection map GUI')
        self.setGeometry(750, 100, 800, 600)

    @pyqtSlot()
    def on_click(self):
        print('load')
        step = int(self.textbox.text())
        feat13_arr = []
        feat23_arr = []
        num_cases_arr = []
        for im_file in image_file:
            feat13, feat23, num_cases = test_moving_window(model_file, im_file, step=step)
            feat13_arr.append(feat13)
            feat23_arr.append(feat23)
            num_cases_arr.append(num_cases)
        test_plane_graph(feat13_arr, feat23_arr, num_cases_arr, step)
        plt.show()

    def on_click2(self):
        step = int(self.textbox.text())
        test_plane_stats(model_file, step)
        plt.show()


class Figure_Canvas(QWidget):

    def __init__(self, parent):
        super().__init__(parent)

        self.setAcceptDrops(True)

        blabla = QLineEdit()

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(blabla)

        self.buttonLayout = QWidget()
        self.buttonLayout.setLayout(self.right_layout)

    def dragEnterEvent(self, e):

        if e.mimeData().hasFormat('text/uri-list'):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        global image_file, model_file
        for url in e.mimeData().urls():
            path = url.toString()
            ext = path.split('.')[-1]
            path = path.replace('file://','')
            if ext in ['tif', 'tiff', 'png']:
                image_file.append(path)
                print('Image loaded {}'.format(image_file))
            if ext in ['pt', 'model']:
                model_file = path
                print('Model loaded {}'.format(model_file))


if __name__ == '__main__':

    # GUI to test one model on one image
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    app.exec_()

    # Test all models for regression
    test_everything()

    # Export noisy figures
    print_noisy()

    # Test all models for noise degradation
    test_noise()

    # Test plane depth detection and output graph
    test_plane_stats(model_file=None, step=128)