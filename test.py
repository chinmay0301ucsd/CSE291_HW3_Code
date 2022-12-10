import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
# from utils.visualization import render_poses, poses_to_render
import argparse 
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


def test(params):
	assert os.path.exists(params['loadpath']), "model path doesn't exist"
	if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
		os.makedirs(os.path.join(params['savepath'], params['exp_name']))
	
	savepath = os.path.join(params['savepath'], params['exp_name'])
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'], params['res_factor'])
	cam_params = rg.cam_params
	net = Nerf().cuda()
	net.load_state_dict(torch.load(params['loadpath']),strict=True)
	
	if params['animation']:
		theta = -params['theta']
		n_phi = params['num_poses']
		poses = poses_to_render(r=4, theta=theta, n_phi=n_phi)
		render_poses(net, poses, cam_params, batch_size, savepath,\
					N=params['Nf'], tn=params['tn'], tf=params['tf'])
		return

	im_set = params['im_set'] # can be 'train', 'test', 'val' depending on whcich images to render
	print(f"saving images to {params['savepath']}")
	avg_psnr = 0
	idxs = params['im_idxs'] if type(params['im_idxs']) is list else list(range(len(rg.samples[im_set])))

	with torch.no_grad():
		for idx in idxs:
			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=params['batch_size'],\
											 		  im_idx=idx, im_set=im_set, N=params['Nf'] + params['Nc'],\
													  tn=params['tn'], tf=params['tf'])
			avg_psnr += img_psnr(gt_img, rgb_img)

			rgb_out = torch.cat((torch.from_numpy(gt_img), rgb_img),axis=0)
			save_image(make_grid(rgb_out.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'rgb_{idx}.png'))
			save_image(make_grid(depth_img.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'depth_{idx}.png'))
		avg_psnr = avg_psnr / len(idxs)
		print(f"Average PSNR : {avg_psnr}")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='/home/ubuntu/NeRF_CT/configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	test(params['test_params'])


