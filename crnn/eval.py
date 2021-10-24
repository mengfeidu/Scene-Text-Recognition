import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import params
import argparse
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
# parser.add_argument('-i', '--image_path', type = str, required = True, help = 'demo image path')
# args = parser.parse_args()

model_path = 'expr_all/netCRNN_188_1241.pth'
# image_path = 'train_data/img/2.jpg'

temp_img = 'temp.jpg'
result_path = './result_222/'
os.mkdir(result_path)


# net init
nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(params.alphabet)

transformer = dataset.resizeNormalize((100, 32))

def eval(image_path):
    image = Image.open(image_path).convert('L')
    image = transformer(image)
    # if torch.cuda.is_available():
    image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred
#########################################################################
import numpy as np
import cv2 as cv


gt_path = '../submit'
gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]

test_img_path = '../dataset/test/img/'
test_img_files = [gt_file for gt_file in sorted(os.listdir(test_img_path))]

def extract_vertice(path):
    vertices = []
    labels = []
    with open(path, 'r',encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        a = np.array(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        a = a.reshape((4,2))
        a = np.float32(a)
        vertices.append(a)
        # label = line.rstrip('\n').lstrip('\ufeff').split(',')[-1]
        # labels.append(label)
    return vertices, lines

def get_lan_img(img_org, v):
    img_org = cv.imread(img_org, 1)
    x = int(np.sum((v[0] - v[1]) ** 2) ** 0.5) + 1
    y = int(np.sum((v[1] - v[2]) ** 2) ** 0.5) + 1
#     print(x, y)
    new_img = np.zeros(shape=[x, y, 3])
    points1 = v
    points2 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
    # 变换矩阵M
    M = cv.getPerspectiveTransform(points1, points2)
    # 变换后的图像
    processed = cv.warpPerspective(img_org, M, (x, y))
    return processed




# ################### fot test
# test_img_path = './'
# test_img_files = ['tr_img_00001.jpg']
# gt_files = ['emp.txt']
# ####################
for i, test_img in enumerate(test_img_files):
    print(i)
    gt = gt_files[i]
    vertices, rawlines = extract_vertice(gt)
    name = test_img[:-4]
    img_pth = test_img_path+test_img
    print(img_pth)

    if len(rawlines) == 0:
        with open(result_path+name+'.txt', 'w', encoding='utf-8') as f:
            pass
    else:
        # all_results = []
        for j, vertice in enumerate(vertices):
            rawline = rawlines[j]
            propcessed = get_lan_img(img_pth, vertice)
            cv.imwrite(temp_img, propcessed)
            txt = eval(temp_img)
            # all_results.append(rawline+','+txt)
            with open(result_path+name+'.txt', 'a', encoding='utf-8') as f:
#                 print(rawline[:-1]+','+txt+'\n')
                f.write(rawline[:-1]+','+txt+'\n')











