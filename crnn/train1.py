from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as net
import params1
import imp
imp.reload(dataset)
imp.reload(params1)
params = params1
# parser = argparse.ArgumentParser()
# parser.add_argument('-train', '--trainroot', required=True, help='path to train dataset')
# parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
# args = parser.parse_args()

train_img = './train_data/img'
train_label = './train_data/label'


val_img = './val/img'
val_label = './val/label'


if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
def data_loader():
    # train
    train_dataset = dataset.lmdbDataset(train_img, train_label)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize,
                                               shuffle=True, sampler=sampler, num_workers=int(params.workers),
                                               collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    
    # val
    val_dataset = dataset.lmdbDataset(val_img, val_label, transform=dataset.resizeNormalize((params.imgW, params.imgH)))
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return train_loader, val_loader

train_loader, val_loader = data_loader()

# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    
    return crnn

crnn = net_init()
#print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
if params.sgd:
    optimizer = optim.SGD(crnn.parameters(), lr=params.lr, momentum = 0.9,weight_decay = 1e-4)
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy


def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # print(cost)
    # crnn.zero_grad()
    if str(cost.item()) != 'inf' and str(cost.item()) != 'nan':
        cost.backward()
        optimizer.step() 
#         print('train!')
#     cost.backward()
#     optimizer.step() 
    return cost

import matplotlib.pyplot as plt

def plot_loss_diffopt(loss_list, acc_list):
    color = ['blue', 'green','red', 'yellow']
    index = 0
    step = list(range(1,len(loss_list)+1))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(step, loss_list, color=color[0], alpha=0.5, label='loss value')
    ax2.plot(step, acc_list, color=color[1], alpha=0.5, label='acc')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss value")
    ax2.set_ylabel("accuracy rate")
    ax2.set_ylim((0, 1))
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    
    plt.show()
    plt.savefig('p3.png')



acc_list = []
loss_list = []

if __name__ == "__main__":
    best_acc = 0
    best_epoch = 0
    for epoch in range(params.nepoch):
#         if epoch % params.valInterval == 0:
#             val(crnn, criterion)
        
#        epoch += 0
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            cost = train(crnn, criterion, optimizer, train_iter)
            if str(cost.item()) != 'inf' and str(cost.item()) != 'nan':
                loss_avg.add(cost) 
            
            i += 1
            
            if i % 500 == 0:
                print(cost.item())
        loss_list.append(loss_avg.val())
        if epoch % params.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if epoch % params.valInterval == 0:
            acc = val(crnn, criterion)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            print("best acc= ",best_acc,'----- best epoch= ',best_epoch)
        acc_list.append(acc)
        
            # do checkpointing
        if (epoch+1) % params.saveInterval == 0:
            torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))
            print('save')
        if (epoch+1) % 5 == 0:
            plot_loss_diffopt(loss_list, acc_list)