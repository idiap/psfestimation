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

import torch.nn as nn
from pandas.io.parsers import read_csv
from torch import FloatTensor as Tensor
import glob
import torch.utils as utils
from torch.autograd import Variable
import io as _io
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import folder
from skimage import io
import os
from toolbox import *

def grayloader(path):
    '''
    Image loader
    '''
    img = io.imread(path, as_gray=True)
    return img


class Noise(object):
    '''
    Ads noise to a pytorch image
    '''
    def __init__(self, probability, noise_type):
            self.probabilit = probability
            self.noise_type = noise_type
    def __call__(self, img):
        img = Tensor(img)
        img = img[None, :,:]
        if self.noise_type is None:
            return img
        if self.probabilit > 0.0:
            img /= 65535.0
            img = noisy(img, self.noise_type, self.probabilit)
            if (img.max() > 1.0):
                img /= img.max()
            if (img.min() < 0):
                img -= img.min()
            if (img.max() > 1.0):
                img /= img.max()
            img *= 65535.0
        return img


class To_224(object):
    '''
    Upscale image to 224x224x3
    '''
    def __init__(self):
        pass
    def __call__(self, img):
        img = img.repeat(3, 1, 1)
        img.unsqueeze_(0)
        m = nn.Upsample(size=(224,224), mode='bilinear')
        img = m(img)
        return img[0]


current_epoch = 0


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(utils.data.Dataset):

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class DatasetFromHdf5(utils.data.Dataset):
    def __init__(self, data_file_name, loader =None, transform=None, target_transform=None):

        print('{}'.format(data_file_name))
        hf = h5py.File(data_file_name, mode='r')
        self.data = hf['data']
        self.labels = hf['labels']
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform

    def __getitem__(self, index):


        while True:
            try:
                image = self.data[index]
                sample = self.loader(_io.BytesIO(image.tostring()))
                target = self.labels[index]
                sample = sample.astype(np.float)

                break
            except:
                index = index - 1


        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample.float(), target

    def __len__(self):
        return self.data.shape[0]

class ImageFolder(DatasetFolder):

    def __init__(self, root, file_list=None, transform=None, target_transform=None,
                 loader=folder.default_loader):
        super(ImageFolder, self).__init__(root, loader, folder.IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        self.file_list = file_list

    def __getitem__(self, index):
        while True:
            try:
                path, _ = self.samples[index]
                idx_extracted = int(path[-13:-4])
                target = self.file_list[idx_extracted]
                sample = self.loader(path)
                break
            except:
                index = rand_int(-10,10)
        if target[1] == 1000.:
            target[1] = 10.
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample.float(), target


def load_crops(folder_prefix = '', test = False, patch_size=128, synthetic=10, natural=1, points=10, black=10, model_type='2dgaussian', batch_size=8, noise = 0.0, isnew = False, noise_type=None, suffix=''):
    global num_classes, log

    if test:
        log.info("Loading test set...")
    else:
        log.info("Loading train set...")

    if model_type == '1dgaussian':
        num_classes = 2
    elif model_type == '2dgaussian':
        num_classes = 3
    elif model_type == '1dzernike':
        num_classes = 2
    elif model_type == '2dzernike':
        num_classes = 4
    elif model_type == 'astzernike':
        num_classes = 3
    elif model_type == 'astsphzernike':
        num_classes = 4
    else:
        print('Undefined model type')
        exit()


    if patch_size != 128:
        transform = transforms.Compose([Noise(probability=noise, noise_type=noise_type), To_224()])
        patch_size = 128
    else:
        transform = transforms.Compose([Noise(probability=noise, noise_type=noise_type)])

    if test and isnew:
        folder_name = '{}/psf_{}_n_{}_s_{}_p_{}_b_{}{}_{}_0_test/'.format(folder_prefix, patch_size, natural, synthetic, points, black, suffix, model_type)
    elif test:
        folder_name = '{}/psf_{}_n_{}_s_{}_p_{}_b_{}{}_{}_test/'.format(folder_prefix, patch_size, natural, synthetic, points, black, suffix, model_type)
    else:
        folder_name = '{}/psf_{}_n_{}_s_{}_p_{}_b_{}{}_{}_train/'.format(folder_prefix, patch_size, natural, synthetic,
                                                                       points, black, suffix, model_type)

    _file_csv = read_csv(os.path.expanduser(folder_name+"parameters.txt"))
    _header = _file_csv.head(0).columns.base

    #_set = ImageFolder(folder_name, transform=transforms.ToTensor(),
    #                               loader=grayloader, file_list=_file)

    _set = DatasetFromHdf5(folder_name+"data.h5", loader = grayloader,  transform=transform)
    if not test:
        _loader = torch.utils.data.DataLoader(dataset=_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    else:
        _loader = torch.utils.data.DataLoader(dataset=_set, batch_size=batch_size, shuffle=True,  num_workers=0)

    if noise_type is not None:
        for test_images, test_labels in _loader:
            image = test_images[0][0].data.cpu().numpy()
            io.imsave('noise_sample_{}_{:.3f}.png'.format(noise_type, noise), image/65535.0)
            break

    return _loader, _header


def load_oneimage(size=64, filename='image.tif'):
    global patch_size, batch_size
    patch_size = size
    image = grayloader('image.tif')
    train_loader = torch.utils.data.DataLoader(dataset=image, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=image, batch_size=1, shuffle=True)
    return train_loader, test_loader

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs).cuda()
    name = "resnet18"
    return model, name

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs).cuda()
    name = "resnet34"
    return model, name

class BasicBlockX(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlockX, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], num_group, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlockX, [2, 2, 2, 2], **kwargs).cuda()
    name = "resnext18"
    return model, name


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlockX, [3, 4, 6, 3], **kwargs).cuda()
    name = "resnext34"
    return model, name


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(BottleneckX, [3, 4, 6, 3], **kwargs).cuda()
    name = "resnext50"
    return model, name


def resnet34_pretrained():
    model = torchvision.models.resnet34(pretrained=True)
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, num_classes)
    model = model.cuda()
    return model, 'resnet34pretrained'


def resnet50_pretrained():
    model = torchvision.models.resnet50(pretrained=True)
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, num_classes)
    model = model.cuda()
    return model, 'resnet50pretrained'


def l2loss(outputs, labels):
    zero_or_one = (1.0 - labels[:,0])
    loss_flag =  ((outputs[:,0] - labels[:,0])**2).mean()
    loss_parameters = ((outputs - labels)**2).mean(1)
    loss = (zero_or_one * loss_parameters).mean() + loss_flag
    return loss

def l2variance(outputs, labels):
    zero_or_one = (1.0 - labels[:,0])
    loss_flag =  ((outputs[:,0] - labels[:,0])**2)
    loss_parameters = ((outputs - labels)**2)
    loss = (zero_or_one * loss_parameters.mean(1)) + loss_flag
    loss_average = loss.mean()
    variance = ((loss-loss_average)**2).mean()
    return variance

def train(run_name, train_loader, test_loader):
    global log
    model.train(True)
    global niter, current_epoch
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    test_errors = []
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.9 ** (epoch // 2))
        return lr

    for ep in range(current_epoch, epoch):
        current_epoch = ep
        log.info("Starting to train Epoch {}".format(ep))
        batch_nb = 0
        cum_loss = 0
        lr = adjust_learning_rate(optimizer, ep)
        log.info("Learning rate : {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        image_num = 0
        for image, label in train_loader:
            image = Variable(image).cuda()
            optimizer.zero_grad()
            output = model(image)
            loss = l2loss(output, Variable(label.float()).cuda())
            loss.backward()
            optimizer.step()

            niter = ep * len(train_loader) + batch_nb

            cum_loss += loss.cpu().data.numpy()
            batch_nb += 1
            image_num += output.size(0)
            average_error = np.around(cum_loss / batch_nb, 3)
            log.info(
                "Ep {0}/{1}, lr {6:.1E}, bt {2}/{3}, loss {4:.2E}, avg loss {5:.2E}".format(ep, epoch, batch_nb, len(train_loader),
                                                                                np.around(loss.cpu().data.numpy(), 3),
                                                                                      average_error, lr))
            if len(test_errors) > 0:
                log.info("Test errors : {}".format(test_errors[-1]))

        err = test_image(model, test_loader)
        save("{0}_{1}_ep{2:0>2}_trainerr{3:.2}_testerr{4:.2}".format(run_name.strip("/"), model_name, current_epoch, average_error, err))
        test_errors.append(err)
    log.info("The training is complete.")


def save(run_name):
    log.info("Saving the model...")
    torch.save(model, 'model_{}.pt'.format(run_name.strip("/")))


def load(run_name):
    log.info("Loading the model {}... ".format(run_name))
    model = torch.load('model_{}.pt'.format(run_name.strip("/")))
    return model


def test_image(model, loader, max_image=1000):
    global log
    model.eval()
    for child in model.children():
        if type(child) == nn.BatchNorm2d:
            child.track_running_stats = False
    i = 0
    nb_batch = 0
    cumloss = 0

    for image, label in loader:
        if i > max_image:
            break
        output = model(Variable(image, requires_grad=False).cuda())
        loss = l2loss(output, Variable(label).cuda().float())
        output = (output*1000.).round() / 1000.
        output[:,0] = output[:,0].round()
        i += output.size(0)
        nb_batch += 1
        cumloss += loss.cpu().data.numpy()
        error_average = cumloss/nb_batch
        log.info("Batch {}/{}, error : {}, avg error {}".format(nb_batch,len(loader),loss.cpu().data.numpy(), error_average))

    log.info("error on the full set : {}".format(error_average))

    return error_average

import logging

log = logging.getLogger('')
log.setLevel(logging.INFO)

if __name__ == '__main__':

    epoch = 20
    learning_rate = 0.00001

    folder_prefix = "data"
    batch_size = 32

    patch_size = 128
    synthetic = 0
    natural = 1
    points = 0
    black = 5
    model_type = '1dzernike'
    noise = 0.0
    noise_type = 'poisson'
    model_name = 'resnet34'
    suffix = '_noise_0'
    run_nb = 201
    run_name = '{}_n_{}_s_{}_p_{}_b_{}{}_{}_{}/'.format(patch_size, natural, synthetic, points, black, suffix, model_type, run_nb)

    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name),
        handlers=[
            logging.FileHandler("output_log_{}.log".format(run_nb)),
            logging.StreamHandler()
        ])

    niter = 0
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    test_loader,test_header = load_crops(folder_prefix= folder_prefix, test=True, patch_size=patch_size, synthetic=synthetic, natural=natural, points=points, black=black, model_type=model_type, batch_size=16, noise= noise, isnew=False, noise_type=noise_type, suffix= suffix)


    train_loader,train_header = load_crops(folder_prefix= folder_prefix, test=False, patch_size=patch_size, synthetic=synthetic, natural=natural, points=points, black=black, model_type=model_type, batch_size=batch_size, noise=0.0, suffix= suffix)

    log.info("Starting model run {}....".format(run_name))

    st = "model_{0}_{1}*".format(run_name.strip("/"), model_name)
    list_saves = glob.glob(st)
    if len(list_saves) > 0:
        list_saves = sorted(list_saves)
        idx = list_saves[-1].find("_ep")+3
        current_epoch = int(list_saves[-1][idx:idx+2])
        log.info("Loading file {}... epoch {}".format(list_saves[-1], current_epoch))
        model = torch.load(list_saves[-1])
    elif model_name == 'resnet34':
        model, model_name = resnet34()
    elif model_name == 'resnext50':
        model, model_name = resnext50()
    elif model_name == 'resnet34pretrained':
        model, model_name = resnet34_pretrained()
    elif model_name == 'resnet50pretrained':
        model, model_name = resnet50_pretrained()
    else:
        log.error("MODEL {} NOT FOUND".format(model_name))


    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name+model_name),
        handlers=[
            logging.FileHandler("output_log_{}.log".format(run_nb)),
            logging.StreamHandler()
        ])

    train(run_name, train_loader, test_loader)