import numpy as np
import torch 
import cv2 
import torch
import os 
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from utils.xyz import positional_encoder
from utils.xyz import rays_single_cam 
from utils.rendering import volume_render

def sample_pdf(ts, weights, Nf):
  ## create cdf from weights

  weights += 1e-6 # ensures non zero denominator in cdf 
  cdf = torch.cumsum(weights / torch.sum(weights,dim=-1,keepdims=True), dim=-1)
  cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
  Nc, Nt = cdf.shape

  ## samples N_sample random numbers 
  u = torch.rand([Nc, Nf])

  ## inverse transform sampling 
  u = u.contiguous()
  idxs = torch.searchsorted(cdf, u, right=True)
  upper = torch.min((Nt - 1)*torch.ones_like(idxs), idxs)
  lower = torch.max(torch.zeros_like(idxs), idxs-1)
  idxs_gather = torch.stack([lower, upper], -1)  # (Nc, N_samples, 2)

  ## generate new t values for sampling in between 
  gather_shape = [Nc, Nf, Nt]
  cdf_gather = torch.gather(cdf.unsqueeze(1).expand(gather_shape), 2, idxs_gather)
  ts_gather = torch.gather(ts.unsqueeze(1).expand(gather_shape), 2, idxs_gather)

  ## generating new sample  values
  t = (u-cdf_gather[:,:,0]) # New samples
  denominator = (cdf_gather[:,:,1]-cdf_gather[:,:,0])
  denominator = torch.where(denominator < 1e-6, torch.ones_like(denominator), denominator)
  t /= denominator
  samples = ts_gather[:,:,0] + t*(ts_gather[:,:,1]-ts_gather[:,:,0])
  return samples

def render_nerf_Hier(rays, netC, netF, Nc, Nf, tn, tf ):
	rgb_c, depth_c, alpha_c, acc_c, w_c, ts = render_nerf_ts(rays, netC, Nc, tn, tf)
	ts_mid = (ts[:,1:] + ts[:,:-1])/2
	ts_hier = sample_pdf(ts_mid, w_c[:,1:-1], Nf)
	ts_hier = ts_hier.detach()

	ts_final, _ = torch.sort(torch.cat([ts, ts_hier], -1), -1)
	B = rays.size(0)
	origins = rays[:,:3] # Bx3
	dirs = rays[:,3:]  # Bx3

	disp = dirs.unsqueeze(-1)*ts_final.unsqueeze(1) # Bx1x3 * Bx(Nc+Nf)x1 = Bx3x(Nc + Nf)
	
	locs = origins.unsqueeze(-1) + disp # Bx3x1 + Bx3xN = Bx3xN 
	dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)	
	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,Nc + Nf)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)
	out = netF.forward(query_pts)
	out = out.reshape(B,Nc + Nf,4)
	rgb_f, depth_f, alpha_f, acc_f, w_f = volume_render(out, ts_final, dirs)

	return (rgb_c, rgb_f), (depth_c, depth_f), (alpha_c, alpha_f), (acc_c, acc_f), (w_c, w_f)

def render_image_Hier(netC, netF, rg, batch_size=64000, im_idx=0, im_set='val', Nc=128, Nf=64, tn=2, tf=6):
	""" render an image and depth map from val set (currently hardcoded) from trained NeRF model
	batch_size: batch size of rays 
	N: number of samples per ray 
	tn: lower limit of distance along ray 
	tf: upper limit of distance along ray
	"""
	gt_img = rg.samples[im_set][im_idx]['img']
	H,W = gt_img.shape[0], gt_img.shape[1]
	NUM_RAYS = H*W 
	netC = netC.cuda()
	netF = netF.cuda()
	rays = rg.rays_dataset[im_set][im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS,:]
	rgbs_c = []
	rgbs_f = [] 
	depths = [] 
	with torch.no_grad():
		for i in tqdm(range(rays.size(0) // batch_size)):
			inp_rays = rays[i*batch_size:(i+1)*batch_size]
			(rgbc, rgbf), (depthc, depthf), _, _, _ = render_nerf_Hier(inp_rays.cuda(), netC, netF, Nc=Nc, Nf=Nf, tn=tn, tf=tf)
			rgbc = torch.clip(rgbc, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
			rgbf = torch.clip(rgbf, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
			depth = torch.clip(depthf, torch.tensor(tn).cuda(), torch.tensor(tf).cuda())
			depth = depthf / torch.max(depthf)
			rgbs_c.append(rgbc)
			rgbs_f.append(rgbf)
			depths.append(depth)

	rgb_c = torch.cat(rgbs_c).cpu()
	rgb_f = torch.cat(rgbs_f).cpu()
	depth = torch.cat(depths).cpu()
	
	rgb_c_img = rgb_c.reshape(1,H,W,3) ## permuting for tensorboard
	rgb_f_img = rgb_f.reshape(1,H,W,3)
	depth_img = depth.reshape(1,H,W,1) ## permuting for tensorboard
	gt_img = gt_img.reshape(1,H,W,3)
	return (rgb_c_img, rgb_f_img), depth_img, gt_img

def render_nerf_ts(rays, net, N, tn=2, tf=6):
	""" same as render nerf but also returns ts"""
		## input to nerf model BN x 6 
	## Nerf output - BN x 4  --> reshape BxNx4 
	## t sample dims - BxN
	B = rays.size(0)
	t_bins = torch.linspace(tn,tf,N+1, device='cuda:0')
	bin_diff = t_bins[1] - t_bins[0] 

	unif_samps = torch.rand(rays.size(0),N, device='cuda:0')
	ts = bin_diff* unif_samps + t_bins[:-1] # BxN 
	# ts = ts.cuda()
	origins = rays[:,:3] # Bx3
	dirs = rays[:,3:]  # Bx3

	disp = dirs.unsqueeze(-1)*ts.unsqueeze(1) # Bx1x3 * BxNx1 = Bx3xN
	
	locs = origins.unsqueeze(-1) + disp # Bx3x1 + Bx3xN = Bx3xN 
	dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)	
	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,N)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)
	out = net.forward(query_pts)
	out = out.reshape(B,N,4)
	rgb_v, depth_v, alpha_v, acc_v, w_v = volume_render(out, ts, dirs)

	return rgb_v, depth_v, alpha_v, acc_v, w_v, ts
