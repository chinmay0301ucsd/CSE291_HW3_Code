import numpy as np
import math 
import random
import os 
import glob
import json
import cv2
import re 
import torch
from natsort import natsort_keygen, ns
from utils.xyz import rays_single_cam
from utils.phaseoptic import PhaseOptic, unif_lenslet_params, raytrace_phaseoptic


def load_hw3_data(path, res_factor=0.25, num_imgs=-1, combine_train_val=False):
	""" load cse 291 homework 3 data """
	## scene bounding volume 
	bbox_path = os.path.join(path, 'bbox.txt')
	bbox = np.loadtxt(bbox_path)[:-1]
	xmin, ymin, zmin, xmax, ymax, zmax = bbox

	## ground truth poses and images 
	pose_paths = sorted(glob.glob(os.path.join(path, 'pose/*')))
	img_paths = sorted(glob.glob(os.path.join(path, 'rgb/*')))
	
	## Temporary thing to check effect of val vs train poses
	train_pose_paths, train_img_paths = pose_paths[:100], img_paths[:100]
	# val_pose_paths, val_img_paths = pose_paths[:num_imgs], img_paths[:num_imgs]
	val_pose_paths, val_img_paths = pose_paths[100:200], img_paths[100:200]
	test_pose_paths= pose_paths[200:]

	intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt')).astype(np.int32)
	cam_params = [intrinsics[0,2]*2, intrinsics[1,2]*2, intrinsics[0,0]]


	if num_imgs < 0:
		num_imgs = len(train_img_paths)

	keys = ['train', 'test', 'val']
	if not combine_train_val:
		img_path_dict = {'train': train_img_paths, 'val':val_img_paths}
		pose_path_dict = {'train': train_pose_paths, 'test':test_pose_paths, 'val':val_pose_paths}
	else:
		img_path_dict = {'train': train_img_paths + val_img_paths, 'val':val_img_paths}
		pose_path_dict = {'train': train_pose_paths + val_pose_paths, 'test':test_pose_paths, 'val':val_pose_paths}
		print("combining train and val sets for better results")
		print(len(img_path_dict['train']))
		print(len(pose_path_dict['train']))
	samples = {'train':[], 'test':[], 'val': []} 

	for i in range(200):
		transform = torch.from_numpy(np.loadtxt(pose_path_dict['test'][i])).float()
		samples['test'].append({'transform':transform, 'img': np.zeros((int(800*res_factor), int(800*res_factor), 3))})

	for key in ['train', 'val']:
		for i in range(num_imgs):
			train_img = cv2.cvtColor(cv2.imread(img_path_dict[key][i]), cv2.COLOR_BGR2RGB) / 255.0
			transform = torch.from_numpy(np.loadtxt(pose_path_dict[key][i])).float()
			H,W = train_img.shape[:2]
			train_img = cv2.resize(train_img, (int(W*res_factor) , int(H*res_factor)), interpolation=cv2.INTER_AREA)
			samples[key].append({'img': train_img, 'transform':transform})

	cam_params = [ele*res_factor for ele in cam_params]
	cam_params[0], cam_params[1] = int(cam_params[0]), int(cam_params[1])
	return samples, cam_params

def load_data(path, half_res=True, num_imgs=-1):
	"""
	Assume path has the following structure - 
	path/ -
	  test/
	  train/
	  val/
	  transforms_test.json
	  transforms_train.json
	  transforms_val.json

	Assumes that frames are ordered in the json files 

	Returns:
	  samples {'train':train, 'test': test, 'val': val}
	  cam_params [H, W, f]
	"""

	train_path = os.path.join(path, 'train')
	test_path = os.path.join(path, 'test') 
	val_path = os.path.join(path, 'val')
	
	sk = natsort_keygen(alg=ns.IGNORECASE)

	train_img_paths = glob.glob(os.path.join(train_path,'*'))
	val_img_paths = glob.glob(os.path.join(val_path,'*'))
	test_img_paths = [os.path.join(test_path,fname) for fname in os.listdir(test_path) if re.match(r"r_[0-9]+.png", fname)]
	test_depth_paths = glob.glob(os.path.join(test_path,'r_*_depth*'))
	test_normal_paths = glob.glob(os.path.join(test_path, 'r_*_normal*'))

	train_img_paths.sort(key=sk)
	val_img_paths.sort(key=sk)
	test_img_paths.sort(key=sk)
	test_depth_paths.sort(key=sk)
	test_normal_paths.sort(key=sk)
	
	with open(os.path.join(path, 'transforms_train.json')) as f:
		train_transform = json.load(f)
	with open(os.path.join(path, 'transforms_test.json')) as f:
		test_transform = json.load(f)
	with open(os.path.join(path, 'transforms_val.json')) as f:
		val_transform = json.load(f)

	if num_imgs < 0:
		num_train = len(train_img_paths)
		num_val = len(val_img_paths)
		num_test = len(test_img_paths)
	else:

		num_train = num_val = num_test = num_imgs

	## generate training samples 
	train_samples = []
	for i in range(num_train):
		train_img = cv2.cvtColor(cv2.imread(train_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = train_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = train_img.shape[:2]
			train_img = cv2.resize(train_img, (W//2 , H//2 ), interpolation=cv2.INTER_AREA)
		train_samples.append({'img': train_img, 'transform':transform, 'metadata':metadata})

	## generate val samples 
	val_samples = [] 
	for i in range(num_val):
		val_img = cv2.cvtColor(cv2.imread(val_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = val_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = val_img.shape[:2]
			val_img = cv2.resize(val_img, (W//2, H//2), interpolation=cv2.INTER_AREA)

		val_samples.append({'img': val_img, 'transform':transform, 'metadata':metadata})
	

	test_samples = [] 
	for i in range(num_test):
		img = cv2.cvtColor(cv2.imread(test_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_depth = cv2.cvtColor(cv2.imread(test_depth_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_normal = cv2.cvtColor(cv2.imread(test_normal_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = test_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = img.shape[:2]
			img = cv2.resize(img, (W//2 ,H//2), interpolation=cv2.INTER_AREA)

		test_samples.append({'img': img, 'img_depth': img_depth, 'img_normal':img_normal,\
			 				 'transform':transform, 'metadata':metadata})	

	## calculate image params and focal length 
	fov = train_transform['camera_angle_x']
	H, W = img.shape[:2]
	f = W /(2 * np.tan(fov/2))
	cam_params = [H,W,f]
	

	## TODO: Implement half res image loading 
	samples = {} 
	samples['train'] = train_samples
	samples['test'] = test_samples
	samples['val'] = val_samples
	return samples, cam_params   

def rays_dataset(samples, cam_params, phase_optic=None):
	""" Generates rays and camera origins for train test and val sets under diff camera poses""" 
	keys = ['train', 'test', 'val']
	rays_1_cam = rays_single_cam(cam_params)
	if phase_optic is not None:
		out = raytrace_phaseoptic(cam_params, phase_optic)
		_,_, rays_phaseop = out['rays_trace']
		rays_phaseop = torch.from_numpy(rays_phaseop).t().float()
	rays = {}
	cam_origins = {}
	H, W, f = cam_params
	for k in keys:
		num_images = len(samples[k])
		transf_mats = torch.stack([s['transform'] for s in samples[k]])
		if phase_optic is None:
			dirs =  torch.matmul(transf_mats[:,:3,:3], rays_1_cam)
			origins = transf_mats[:,:3,3:]
			origins = origins.expand(num_images,3,H*W) #Bx3xHW
		else:
			origins = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[:3,:]) + transf_mats[:,:3,3:]
			dirs = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[3:,:])
		rays[k] = torch.cat((origins, dirs),dim=1).permute(0,2,1).reshape(-1, 6).cuda()
		 # BHW x 6, number of cameras 

	return rays

class RayGenerator:
	def __init__(self, path, res_factor=1, num_imgs=-1, phase_dict=None, flip_ray_dir=False, combine_train_val=False):
		# samples, cam_params = load_data(path, half_res, num_imgs)
		samples, cam_params = load_hw3_data(path, res_factor, num_imgs, combine_train_val)
		self.samples = samples
		self.cam_params = cam_params
		self.phase_dict = phase_dict
		self.flip_ray_dir = flip_ray_dir
		if phase_dict is not None and self.phase_dict['use_phase_optic']:
			num_lenses = phase_dict['num_lenses']
			radius_scale = phase_dict['radius_scale']
			## currently only uniform lenslets supported 
			centers, radii = unif_lenslet_params(num_lenses,cam_params,radius_scale)
			## generating max over 
			phase_optic = PhaseOptic(centers, radii, mu=1.5)
			self.rays_dataset = rays_dataset(self.samples, cam_params, phase_optic)
		else:
			self.rays_dataset = rays_dataset(self.samples, cam_params)

		if self.flip_ray_dir:
			print("Flipping ray directions")
			for key in self.rays_dataset:
				self.rays_dataset[key][:,3:] *= -1
		
	def select_batch(self, mode='train', N=4000, iter=0):
		""" function for selecting rays non-randomly for training """
		data = self.rays_dataset[mode]
		num_rays = data.size(0)

		rid_min = (N*iter) % num_rays 
		# rid_max = N*(iter+1) % num_rays
		print(rid_min)
		ray_ids = torch.arange(N) + rid_min
		rays = data[ray_ids,:]
		return rays, ray_ids

	def select(self, mode='train', N=4096):
		""" randomly selects N train/test/val rays
		Args:
			mode: 'train', 'test', 'val'
			N: number of rays to sample 
		Returns:
			rays (torch Tensor): Nx6 
			ray_ids: Nx1 
		"""
		data = self.rays_dataset[mode]
		ray_ids = random.sample(range(data.size(0)), N)
		ray_ids = torch.tensor(ray_ids)
	
		# ray_ids = torch.randperm(data.size(0))[:N]
		# print(ray_ids)
		rays = data[ray_ids,:]
		return rays, ray_ids

	def select_imgs(self, mode='train', N=4096, im_idxs=[0,1,2]):
		""" randomly selects N train/test/val rays from a given image
		Args:
			mode: 'train', 'test', 'val'
			N: number of rays to sample
			im_idxs: which image to select
		Returns:
			rays (torch Tensor): Nx6 
			ray_ids: Nx1 
		"""
		NUM_RAYS = self.cam_params[0] * self.cam_params[1]
		data = []
		rays_idxs = [] 
		for im_idx in im_idxs:
			data.append(self.rays_dataset[mode][im_idx*NUM_RAYS:(im_idx + 1)*NUM_RAYS,:])
			rays_idxs.append(np.arange(im_idx*NUM_RAYS, (im_idx + 1)*NUM_RAYS))
		data = torch.cat(data, dim=0)	

		samples = self.samples[mode]
		select_ids = np.random.choice(data.size(0), (N,), replace=False)
		rays_idxs = np.concatenate(rays_idxs)
		rays = data[select_ids, :]
		ray_ids = rays_idxs[select_ids]

		return rays, ray_ids


		
		



