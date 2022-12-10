import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
from utils.hierarchical import *
import argparse 
from torchvision.utils import save_image, make_grid

import yaml 
from tqdm import tqdm 

def img_mse(gt, pred):
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	return torch.mean((pred - gt)**2)

def img_psnr(gt, pred):
	ten = torch.tensor(10)
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	psnr = 20*torch.log(torch.max(gt))/torch.log(ten) - 10 * torch.log(img_mse(gt, pred))/torch.log(ten)
	return psnr 

def train(params):
	if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
		os.makedirs(os.path.join(params['savepath'], params['exp_name']))
	writer = SummaryWriter('logs/run_{}/'.format(str(time.time())[-10:]))
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'], params['res_factor'],
	 				  params['num_train_imgs'],flip_ray_dir=params['flip_ray_dir'],
					  combine_train_val=params['combine_tv'])
	train_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['train']]).reshape(-1,3)
	## exponential Lr decay factor  
	lr_start = params['lr_init']
	lr_end = params['lr_final']

	decay = np.exp(np.log(lr_end / lr_start) / params['num_iters'])
	
	netC = Nerf().cuda()
	netF = Nerf().cuda()
	if params['pretrained_path'] is not None:
		netC.load_state_dict(torch.load(params['pretrained_path']))
		netF.load_state_dict(torch.load(params['pretrained_path']))	
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(list(netC.parameters()) + list(netF.parameters()), lr=lr_start)
	## TODO: Add code to load state dict from pre-trained model
	for i in tqdm(range(params['num_iters'])):
		## main training loopx
		start = time.time()
		rays, ray_ids = rg.select(mode='train', N=batch_size)
		# rays, ray_ids = rg.select_batch(mode='train', N=batch_size, iter=i)
		# rays, ray_ids = rg.select_imgs(mode='train', N=batch_size, im_idxs=[0])
		gt_colors = train_imgs[ray_ids,:].float().cuda()
		optimizer.zero_grad()
		

		(rgb_c, rgb_f),_,_,_, _ = render_nerf_Hier(rays, netC, netF, params['Nc'],\
			 												   params['Nf'], params['tn'], params['tf'])
		# rgbF, depthF, alphaF, accF, wF = render_nerf_fine(rays, netF, wC, params['Nc'], params['Nf'], params['tn'], params['tf'])
		loss_coarse = criterion(rgb_c, gt_colors)
		loss_fine = criterion(rgb_f, gt_colors) 
		loss = loss_coarse + loss_fine
		loss.backward()
		optimizer.step()

		decay_steps = params['decay_step'] * 1000
		lr = lr_start * (params['decay_rate'] ** (i / decay_steps))
		for p in optimizer.param_groups:
			p['lr'] = lr

		## checkpointing and logging code 
		if (i+1) % params['ckpt_loss'] == 0:
			writer.add_scalar("Loss/train", loss.item(), i+1)
			writer.add_scalar("Loss Coarse/train", loss_coarse.item(), i+1)
			writer.add_scalar("Loss Fine/train", loss_fine.item(), i+1)
			writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], i+1)
			print(f'loss coarse: {loss_coarse.item()} | loss fine: {loss_fine.item()} | epoch: {i+1} ')
		
		if (i+1) % params['ckpt_images'] == 0:
			print("--- rendering image ---")
			for ii in params['val_idxs']:
				(rgbc_img, rgbf_img), depth_img, gt_img = render_image_Hier(netC, netF, rg, batch_size=params['render_batch_size'],\
														       im_idx=ii, im_set='train', Nc=params['Nc'], Nf=params['Nf'],\
														       tn=params['tn'], tf=params['tf'])
				writer.add_images(f'train/RGB_C_{ii}', rgbc_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'train/RGB_F_{ii}', rgbf_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'train/Depth_{ii}', depth_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'train/GT_{ii}', gt_img, global_step=i+1, dataformats='NHWC')
				writer.add_scalar(f"Loss/Train_Img_PSNR_F_{ii}", img_psnr(gt_img, rgbf_img), i+1)
				writer.add_scalar(f"Loss/Train_Img_PSNR_C_{ii}", img_psnr(gt_img, rgbc_img), i+1)
				print(f"Train F PSNR: {img_psnr(gt_img, rgbf_img)}")

				(rgbc_img, rgbf_img), depth_img, gt_img = render_image_Hier(netC, netF, rg, batch_size=params['render_batch_size'],\
														  im_idx=ii, im_set='val', Nc=params['Nc'], Nf=params['Nf'],\
														  tn=params['tn'], tf=params['tf'])
				writer.add_images(f'Val/RGB_C_{ii}', rgbc_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'Val/RGB_F_{ii}', rgbf_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'Val/Depth{ii}', depth_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'Val/GT{ii}', gt_img, global_step=i+1, dataformats='NHWC')
				writer.add_scalar(f"Loss/Val_Img_PSNR_F_{ii}", img_psnr(gt_img, rgbf_img), i+1)
				writer.add_scalar(f"Loss/Val_Img_PSNR_C_{ii}", img_psnr(gt_img, rgbc_img), i+1)
				print(f"Val F PSNR: {img_psnr(gt_img, rgbf_img)}")

				
		if (i+1) % params['test_images'] == 0:
			## TODO write function to compute PSNR on all val images.
			for ii in params['test_idxs']:

				(rgbc_img, rgbf_img), depth_img, gt_img = render_image_Hier(netC, netF, rg, batch_size=params['render_batch_size'],\
														  im_idx=ii, im_set='test', Nc=params['Nc'], Nf=params['Nf'],\
														  tn=params['tn'], tf=params['tf'])
				print("saving image of shape: ", rgbc_img.shape)
				writer.add_images(f'test/RGB_C_{ii}', rgbc_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'test/RGB_F_{ii}', rgbf_img, global_step=i+1, dataformats='NHWC')
				writer.add_images(f'test/Depth_{ii}', depth_img, global_step=i+1, dataformats='NHWC')
				save_image(make_grid(rgbc_img.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'rgbc_test_{ii}.png'))
				save_image(make_grid(rgbf_img.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'rgbf_test_{ii}.png'))
				# writer.add_images(f'train/GT_{ii}', gt_img, global_step=i+1, dataformats='NHWC')
				# writer.add_scalar(f"Loss/Train_Img_MSE_{ii}", img_mse(gt_img, rgb_img), i+1)
				# writer.add_scalar(f"Loss/Train_Img_PSNR_{ii}", img_psnr(gt_img, rgb_img), i+1)

		if (i+1)% params['ckpt_model'] == 0:
			print("saving model")
			tstamp = str(time.time())
			torch.save(netC.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'C.pth'))
			torch.save(netF.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'F.pth'))

	print("saving final model")
	tstamp = str(time.time())
	torch.save(netC.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'C.pth'))
	torch.save(netF.state_dict(), os.path.join(params['savepath'], params['exp_name'], tstamp+'F.pth'))

	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='/home/ubuntu/NeRF_CT/configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	train(params)


