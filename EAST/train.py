import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import time
import numpy as np
import re
import matplotlib.pyplot as plt

def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval,checkpoint=None):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
    
	if checkpoint!=None:
		print("Load model %s"%checkpoint)
		start_epoch = int(re.search(r'model_epoch_(\d+).pth',checkpoint).group(1))
		model.load_state_dict(torch.load(checkpoint))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	all_loss = []
	for epoch in range(epoch_iter):	
		model.train()

		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		scheduler.step()
		print('Epoch is [{}/{}], epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch+1, epoch_iter, epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		all_loss.append(epoch_loss/int(file_num/batch_size))
		plt.plot(all_loss)
		plt.savefig('loss_landscape7.png')
		plt.close()
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))



if __name__ == '__main__':

	train_img_path = '../dataset/train/img'
	train_gt_path  = '../dataset/train/gt'
	pths_path      = 'pths'
	batch_size     = 16
	lr             = 1e-3
	num_workers    = 8
	epoch_iter     = 600
	save_interval  = 10
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
	