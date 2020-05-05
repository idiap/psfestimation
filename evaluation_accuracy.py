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

from train import *

def test_moving_window(model_file, image_file, step=128, verbose=True):
    '''
    Output the PSF parameter map from a model and a file. It outputs as well different combinations of parameters for
    depth estimation
    :param model_file: file of the cnn model
    :param image_file: input file
    :param step: moving window step
    :param verbose: output logs
    :return:
    '''
    model = torch.load('{}'.format(model_file))
    num_classes = list(model._modules.items())[-1][1].out_features
    model.train(False)
    for child in model.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
    channels =  list(model._modules.items())[0][1].in_channels
    size = 128
    real_size = size
    x = size
    y = size
    im = io.imread(image_file, as_gray=True)
    if im.dtype == np.uint16:
        im = im.astype(np.int32)

    def To_224(img):
        img = Tensor(img)
        img = img.repeat(3, 1, 1)
        img.unsqueeze_(0)
        m = nn.Upsample(size=(224, 224), mode='bilinear')
        img = m(img)
        return img[0].data.cpu().numpy()

    im = pytoolbox.image.utils.center_crop(im, 50)

    im = im[0:im.shape[0]//size * size, 0:im.shape[1]//size * size]
    weight_image = np.zeros((im.shape[0], im.shape[1], num_classes))

    tile_dataset = []
    y_size = 0
    i = 0

    while x <= im.shape[0]:
        x_size = 0
        while y <= im.shape[1]:
            a = im[x - size:x, y - size:y]

            if channels == 3:
                a = To_224(a)
                real_size = 224
            tile_dataset.append(a[:])
            weight_image[x - size:x, y - size:y] += 1.0
            y += step
            x_size += 1
            i += 1
        y = size
        y_size += 1
        x += step

    tile_dataset = np.asarray(tile_dataset)
    tile_dataset = np.reshape(tile_dataset, (tile_dataset.shape[0], channels, real_size, real_size))

    start = time.time()

    max_size = tile_dataset.shape[0]
    batch_size = 16
    it = 0
    output_npy = np.zeros((tile_dataset.shape[0], num_classes))
    input_tensor = torch.FloatTensor(tile_dataset)

    while max_size > 0:
        num_batch = min(batch_size, max_size)
        out = model(Variable(input_tensor.narrow(0, it, num_batch), requires_grad=False ).cuda())

        out = (out*100.).round() / 100.
        out[:,0] = out[:,0].round()


        output_npy[it:it+num_batch] = out.data.cpu().numpy()
        it += num_batch
        max_size -= num_batch

    end = time.time()
    output = np.zeros((im.shape[0], im.shape[1], output_npy.shape[1]))

    i = 0
    x = size
    y = size
    o = []
    while x <= im.shape[0]:
        while y <= im.shape[1]:
            output[x - size:x, y - size:y] += output_npy[i, :]
            y += step
            i += 1
        y = size
        x += step
    output = output / weight_image
    output[:,:,0] = np.round(output[:,:,0])
    dir = 'output_psfmap'

    print("Time elapsed : ")
    print(end - start)

    if output.shape[2] > 3:
        feat13 = output[:, :, 1] * ((output[:, :, 3]) - 0.5)
        feat23 = ((scale(output[:, :, 3]) - 0.5) * output[:, :, 2])
    else :
        feat13 = output[:, :, 1] * ((output[:, :, 2]) - 0.5)
        feat23 = ((scale(output[:, :, 1]) - 0.5) * output[:, :, 2])
    if verbose:
        plt.figure()
        plt.imshow(im[:, :])
        plt.imsave(dir + '/ast_input.png', im[:, :])

        plt.title('Input')
        for i in range(num_classes):
            plt.figure()
            plt.imshow(output[:, :, i])
            plt.title('Output feat {}'.format(i))

        plt.figure()
        output[:,:,1] = output[:,:,1] / output[:,:,1].max()
        output[:, :, i] = (output[:, :, i] / output[:, :, i].max())
        output[:, :, i-1] = (output[:, :, i-1] / output[:, :, i-1].max())
        combined = output[:, :, i] * output[:, :, i-1]
        combined = combined / combined.max()
        combined -= 0.5
        plt.imshow(output[:, :, 1] * combined)
        plt.title('Output feat 1*2*3')


        plt.figure()
        plt.imshow(feat13)
        plt.title('Output feat 1*3')
        plt.figure()
        plt.imshow(feat23)
        plt.title('Output feat 2*3')
        plt.imsave(dir + '/ast_2and3.png',feat23)

        plt.imsave(dir + '/ast_focus.png', output[:, :, 1] )
        plt.imsave(dir + '/ast_astdirection.png', output[:, :, i] )
        plt.imsave(dir + '/ast_ast.png', output[:, :, i-1] )
        plt.imsave(dir + '/ast_1and3.png', feat13)

    return feat13, feat23, np.sqrt(tile_dataset.shape[0])

def eval_regression_accuracy(model=None, test_loader=None, test_header=None, max_iter = 1000):
    '''
    Estimates the accuration of regression of a model
    :param model: cnn model file
    :param test_loader: image loader for test data
    :param test_header: what are the name of every parameters
    :param max_iter: how many images per model
    '''
    global log

    i = 0
    nb_batch = 0
    cumloss = 0
    cumvariance = 0

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for image, label in test_loader:

            if i >= max_iter:
                break

            img = Variable(image, requires_grad=False).cuda()
            output = model(img)
            output = (output*1000.).round() / 1000.
            output[:,0] = output[:,0].round()

            for a in range(output.size(0)):

                all_predictions.append(output[a].cpu().data.numpy())
                all_labels.append(label[a].data.numpy())

                if label[a,0] == 1:
                    output[a,1:] = 1000.



            loss = l2loss(output, (label).cuda().float())
            variance = l2variance(output, (label).cuda().float())
            cumvariance += variance.cpu().data.numpy()
            cumvariance /= 4
            i += output.size(0)
            nb_batch += 1
            cumloss += loss.cpu().data.numpy()

            log.info("Batch {}/{}, err: {}, avg err: {}, var: {:.3E}, avg var:{:.3E}".format(nb_batch,len(test_loader),loss.cpu().data.numpy(), cumloss/nb_batch, variance.cpu().data.numpy(), cumvariance))

    all_predictions = np.asarray(all_predictions)
    all_labels = np.asarray(all_labels)
    res = {}
    res.update({'error':cumloss/nb_batch, 'variance':cumvariance})

    for feature in range(0, label.shape[1]):
        pred = all_predictions[:,feature]
        lab = all_labels[:,feature]
        new_pred = []
        new_lab = []
        for idx, val in enumerate(lab):
            if val != 1000. and np.abs(lab[idx]- pred[idx]) < 1000.:
                new_lab.append(lab[idx])
                new_pred.append(pred[idx])

        score = r2_score(new_pred, new_lab)
        if test_header[feature] == 'fwmh':
            test_header[feature] = 'focus'
        log.info("Result for feature {} with {} samples, R2 = {}".format(test_header[feature], len(new_pred), score))
        res.update({test_header[feature]:score})

    log.info("Error on the full set: {}".format(cumloss/nb_batch))
    log.info("Error variance on the full set: {}".format(cumvariance))
    return res


if __name__ == '__main__':

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
    run_name = '{}_n_{}_s_{}_p_{}_b_{}{}_{}_{}/'.format(patch_size, natural, synthetic, points, black, suffix, model_type, run_nb)

    import logging

    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name),
        handlers=[
            logging.FileHandler("output_log_{}.log"),
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

    run_name = 2
    model, model_name = load(run_name)

    logging.basicConfig(
        format="%(asctime)s [{}] %(message)s".format(run_name+model_name),
        handlers=[
            logging.FileHandler("output_log_{}.log"),
            logging.StreamHandler()
        ])

    test_moving_window()
    eval_regression_accuracy(model, test_loader, test_header)