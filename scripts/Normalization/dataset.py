"""
AUTHOR: Christopher Z. Eddy (eddyc@ohsu.edu)
Date: 08/08/23
Purpose: Annotations have been made on the DTRON H&E dataset. We would like be able to load 
	individual annotations as input examples for ML/DL training or to obtain morphological
	or textural measurements. This script provides a basic usage to do just that. 

	05/19/23
	Added registration so that each annotation is rotated to be along its first principle component,
	its second axis along the second principle component,
	and centered in the image by the center of mass (COM) if desired. 
	Registration is rotation invariant, verified CE 05/19/23

	MWE provided when calling this script -- please see the bottom of this script
	
	08/08/23
	The strategy of loading a single example at a time may not work. If we randomly select an annotation,
	that could mean loading an image multiple times for just a single batch, and loading the image (albeit in an ome.tif format)
	is the rate-limiting step. Instead, this may be a case where we should consider saving each training
	example. Again, the problem is then we fix the training size and have to redo the analysis again 
	if we wish to use a larger window. And then for cycIF, each image would need to 

	Solution: Holding all the images and annotations actually doesn't increase the memory size significantly,
	since they are stored as dask arrays. Created function: load_all_samples

	Now, we should figure out how many annotations are in each JSON, then the total number of iterables is 
	the summation, but important to split (use cumsum)

UPDATE 11/07/23: The annotations can now be scaled to fit into the fixed input image size. 
	Random crops can also be taken, but in order to make the validation set repeatable, we need to be able to pass a numpy random seed.
"""

import numpy as np
import random
import json 
import glob
import os 
import skimage.draw
import torch
import zarr
import tifffile
import dask.array as da
from dask_image import ndinterp
from xml.etree import ElementTree
#import matplotlib.pyplot as plt 
from scipy import ndimage
from sklearn.decomposition import PCA
#from skimage.transform import rotate 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class GlandDataset(object):
	def __init__(self, config):

		self.config = config

		image_directory = config.dataset_opts['ome_tif_directory']
		annotation_directory = config.dataset_opts['json_annotation_directory']
		#assuming all the data is contained within a directory with subdirectories named "annotations" and "images"
		#the images should be .ome.tif images, generated from our previous scripts. 
		assert os.path.isdir(image_directory), "Path does not exist ::: {}".format(image_directory)
		assert os.path.isdir(annotation_directory), "Path does not exist ::: {}".format(annotation_directory)

		#create image, annotation filepath pairs
		im_names = find_files_by_pattern(image_directory, [".ome.tif"])
		json_names = find_files_by_pattern(annotation_directory, [".json"])

		#for now, exclude any im_names that have 57657 in it, since we are getting permissions sorted.
		#import pdb;pdb.set_trace()
		im_names = [x for x in im_names if "57657.ome.tif" not in x]

		##the number of files in both should be the same.
		#assert len(json_names)==len(im_names), "Found {} files in {}, but only {} in {}".format(len(json_names), annotation_directory, len(im_names), image_directory)

		#now, make the data paired correctly. This may be a user-specific problem, but the easiest solution is the .ome.tif and .json files have the same basename.
		#if this is not the case, you need to write your own code here.

		self.sample_file_pairs = [(os.path.join(image_directory, y), 
								   os.path.join(annotation_directory, [x[:x.rfind(".ome.tif")] for x in [os.path.basename(y)]][0] + '.json')) \
								   for y in im_names]
		
		#now, check that each file name actually exists. 
		arefiles = np.array([[os.path.isfile(x),os.path.isfile(y)] for (x,y) in self.sample_file_pairs])

		#good_inds = [i for i,x in enumerate(np.all(arefiles,axis=1)) if x]
		self.sample_file_pairs = [self.sample_file_pairs[i] for i,x in enumerate(np.all(arefiles,axis=1)) if x]

		assert len(self.sample_file_pairs)>0, "Paired file names do not exist! Check code!"
		print("    Found {} samples to draw examples from...".format(len(self.sample_file_pairs)))
		
		#assert np.all(arefiles.flatten()), "Paired file names do not exist! Check code!"

		print("     Loading all samples...")
		self.load_all_samples() #load all the samples and annotations. 
		print("     Complete.")
		#reindexing each annotation. 
		self.reind = np.cumsum([0]+self.n_annos)[1:] - 1 #n_annos defined in load_all_samples()
		#so now you pick the index by doing index - self.reind, taking the smallest, positive or zero value.


	def load_da_image(self, ome_file_path):
		#utilize lazy loading using dask, zarr, and xarray.
		current_full_image, channel_names = zarr_read(ome_file_path) #set to True for cyc_IF load, use False for H&E. 
		#if you loaded an H&E, the last dimension contains the channels generally.
		if current_full_image.shape[-1] == 3:
			#assume this is RGB channel. 
			current_full_image = da.moveaxis(current_full_image, -1, 0) #move to zeroth axis.
		return current_full_image

	def load_full_sample_anno(self, json_file_path):
		# load the large .json file
		with open(json_file_path) as f:
			current_anno_data=json.load(f)
		return current_anno_data
	
	def load_sample(self, sample_ind: int = 0):
		#self.current_json_file and self.current_ome_file must have been defined. 
		(current_ome_file, current_json_file) = self.sample_file_pairs[sample_ind]
		IM = self.load_da_image(current_ome_file)
		annotation = self.load_full_sample_anno(current_json_file)
		return IM, annotation
		
	def load_all_samples(self):
		self.full_images = []
		self.annotations = []
		for sample_ind in range(len(self.sample_file_pairs)):
			I, anno = self.load_sample(sample_ind=sample_ind)
			
			#return probabilities for selection of an annotation
			if len(anno['annotation'])>1:
				#approx_areas = [(np.stack(x).max(axis=0) - np.stack(x).min(axis=0)).prod() for x in anno['annotation']]
				areas = [PolyArea(x = np.stack(g)[:,0], y = np.stack(g)[:,1]) for g in anno['annotation']]
				#picked_ind = np.argmax(areas)
				probs = [x / sum(areas) for x in areas] #you have a higher probability of selecting the annotation with the larger area, makese sense.
				#whichever has the biggest area, take that one. 
				anno['features']['probs'] = probs
			else:
				anno['features']['probs'] = [1.]
			

			self.full_images.append(I)
			self.annotations.append(anno)
		
		self.n_annos = [len(x['annotation']) for x in self.annotations]
		print("     There are {} samples with {} total annotations.".format(len(self.sample_file_pairs), np.sum(self.n_annos)))

	def load_image_channel(self, x_range: tuple, y_range: tuple, current_full_image: da.Array, channel: int = None):
		#extract the section of the image we want. 
		#after loading the mask, we should also load a subset of coordinates which to pull.
		#use .compute() to load to memory, when needed
		if channel is not None:
			current_im_slice = current_full_image[channel, x_range[0]:x_range[1]-1, y_range[0]:y_range[1]-1]
		else:
			current_im_slice = current_full_image[:, x_range[0]:x_range[1]-1, y_range[0]:y_range[1]-1]
		#self.current_full_image.sel(c=channel, y=slice(int(x_range[0]),int(x_range[1]-1)), x=slice(int(y_range[0]),int(y_range[1]-1)))
		return current_im_slice
	
	def load_instance(self, sample_ind: int = 0, object_n: int = 0, channel: int = None, mask_out: bool = False):
		#load a single annotation from self.current_anno_data
		
		#define the bounding box of that annotation 
		#assuming data is in form of [ [[xvert, yvert], ... ]], [...], ... ] #i.e. a list of lists
		bbox = [int(np.floor(np.min(np.array(self.annotations[sample_ind]['annotation'][object_n])[:,0]))),
				int(np.ceil(np.max(np.array(self.annotations[sample_ind]['annotation'][object_n])[:,0]))),
				int(np.floor(np.min(np.array(self.annotations[sample_ind]['annotation'][object_n])[:,1]))),
				int(np.ceil(np.max(np.array(self.annotations[sample_ind]['annotation'][object_n])[:,1]))),
			]
		
		#now, make the bounding box larger so that if a rotation does occur we can crop it appropriately. 
		# No matter what, it ought to be cropped (see register_annotation for where this occurs)

		if not self.config.dataset_opts['augmentations']['random_crop']:

			# if we are taking the whole gland, we want to do the following:
					
			bbox_width = np.array([bbox[1]-bbox[0], bbox[3]-bbox[2]])
			center = np.array([bbox[0], bbox[2]]) + (bbox_width/2)

			if np.all(bbox_width<=self.config.INPUT_SPATIAL_SIZE):
				for i in range(len(bbox_width)):
					if bbox_width[i] < self.config.INPUT_SPATIAL_SIZE:
						#need to expand to the input spatial size. From the center expand, but also make sure it doesn't extend beyond the image.
						start_ind = max(0, int(np.floor(center[i] - (self.config.INPUT_SPATIAL_SIZE/2))))
						end_ind = min(self.full_images[sample_ind].shape[i+1], start_ind + self.config.INPUT_SPATIAL_SIZE) #the i+1 here is important because the image contains channels in the first dimension 
						if (end_ind - start_ind) != self.config.INPUT_SPATIAL_SIZE:
							#generally, this would happen if the annotation was at the very far edge of the image, which generally doesn't happen
							import pdb;pdb.set_trace()
							start_ind = end_ind - self.config.INPUT_SPATIAL_SIZE
					
						if i==0:
							bbox[0] = start_ind 
							bbox[1] = end_ind 
						elif i==1:
							bbox[2] = start_ind 
							bbox[3] = end_ind
			else:
				for i in range(len(bbox_width)):
					if bbox_width[i] < np.max(bbox_width):
						#need to expand to the input spatial size. From the center expand, but also make sure it doesn't extend beyond the image.
						start_ind = max(0, int(np.floor(center[i] - (np.max(bbox_width)/2))))
						end_ind = min(self.full_images[sample_ind].shape[i+1], start_ind + np.max(bbox_width)) #the i+1 here is important because the image contains channels in the first dimension 
						if (end_ind - start_ind) != np.max(bbox_width):
							#generally, this would happen if the annotation was at the very far edge of the image, which generally doesn't happen
							import pdb;pdb.set_trace()
							start_ind = end_ind - np.max(bbox_width)
					
						if i==0:
							bbox[0] = start_ind 
							bbox[1] = end_ind 
						elif i==1:
							bbox[2] = start_ind 
							bbox[3] = end_ind
		
		else:
			""" TAKE RANDOM CROP WITHIN ANNOTATION """
			#grab the vertices of the annotation, draw a line between any two, and select a pixel along that line to use as the center of the bounding box.

			verts = np.array(self.annotations[sample_ind]['annotation'][object_n])
			#PICK ANY TWO POINTS BETWEEN THE VERTICES AND FORM A LINE, THEN PICK ANY POINT ALONG THAT LINE. IT WILL LIKELY BE WITHIN THE ANNOTION
			picked_inds = np.random.choice(verts.shape[0] - 1, 2, replace=False) #all vertices have the same beginning and ending point. Here, we assure that the first and last index cannot be picked by doing a -1.
			
			picked_verts = verts[picked_inds, :]
			#pick an x anywhere in between the verts.
			try:
				if len(np.unique(picked_verts.astype(int)[:,1]))>1:
					col_ind = np.random.randint(np.sort(picked_verts.astype(int)[:,1])[0], np.sort(picked_verts.astype(int)[:,1])[1])
				else:
					col_ind = picked_verts.astype(int)[0,1]
			except:
				import pdb;pdb.set_trace()
			#calculate the slope and equation of the line separating them.
			if picked_verts[1,1]==picked_verts[0,1]:
				#vertical line.
				row_ind = np.random.randint(np.min(picked_verts[:,0]), np.max(picked_verts[:,0]))
				row_ind = row_ind - self.config.INPUT_SPATIAL_SIZE//2
			else:
				m = (picked_verts[1,0] - picked_verts[0,0])/(picked_verts[1,1] - picked_verts[0,1])
				row_ind = int(np.round((m * col_ind) + picked_verts[1,0] - (m * picked_verts[1,1])))
				row_ind = row_ind - self.config.INPUT_SPATIAL_SIZE//2
			if row_ind < 0:
				row_ind = 0
			col_ind = col_ind - self.config.INPUT_SPATIAL_SIZE//2
			if col_ind < 0:
				col_ind = 0

			#this isn't without flaws, consider that later we want to add a "deltaZ" to the region, which, if we're already at the border, we cannot. Oh well, for now.
			bbox = [row_ind, row_ind + self.config.INPUT_SPATIAL_SIZE, col_ind, col_ind + self.config.INPUT_SPATIAL_SIZE]
			
		
		center = [bbox[0] + (bbox[1] - bbox[0])/2, bbox[2] + (bbox[3] - bbox[2])/2]
		deltaZ = np.sqrt(((bbox[1] - bbox[0])**2) + ((bbox[3] - bbox[2])**2)) #was multiplied by sqrt(2) before, but that shouldn't be necessary.
		#WHAT IS DELTAZ RELATIVE TO BBOX WIDTH?

		x_start = max(0, int(np.floor(center[0] - (deltaZ / 2))))
		x_end = int(np.ceil(x_start + deltaZ))
		if x_end>self.full_images[sample_ind].shape[1]:
			x_end = self.full_images[sample_ind].shape[1]
			x_start = int(np.floor(x_end - deltaZ))
		x_range = (x_start, x_end)
		#x_range = (max(0, int(np.floor(center[0] - (deltaZ / 2)))), min(self.full_images[sample_ind].shape[1], int(np.ceil(center[0] + (deltaZ / 2)))))
		#if x_range[0]>x_range[1]: #only happens if bigger than the image shape!
		#	x_range = (int(np.floor(x_range[1] - deltaZ)), x_range[1])
		y_start = max(0, int(np.floor(center[1] - (deltaZ / 2))))
		y_end = int(np.ceil(y_start + deltaZ))
		if y_end>self.full_images[sample_ind].shape[2]:
			y_end = self.full_images[sample_ind].shape[2]
			y_start = int(np.floor(y_end - deltaZ))
		y_range = (y_start, y_end)
		# y_range = (max(0, int(np.floor(center[1] - (deltaZ / 2)))), min(self.full_images[sample_ind].shape[2], int(np.ceil(center[1] + (deltaZ / 2)))))
		# if y_range[0]>y_range[1]: #only happens if bigger than the image shape!
		# 	y_range = (int(np.floor(y_range[1] - deltaZ)), y_range[1])

		#grab the vertices of the annotation
		verts = np.array(self.annotations[sample_ind]['annotation'][object_n])

		#subtract the mins from verts, so that when we grab the slice the indeces match. 
		
		verts[:,0] = verts[:,0] - int(np.floor(center[0] - (deltaZ / 2)))
		verts[:,1] = verts[:,1] - int(np.floor(center[1] - (deltaZ / 2)))

		#convert to integers.
		verts = np.round(verts).astype(int)

		#grab the image. 
		if channel is not None:
			IM = self.load_image_channel(channel = channel, x_range = x_range, y_range = y_range, current_full_image = self.full_images[sample_ind]) #IM will be H x W
		else:
			IM = da.moveaxis(self.load_image_channel(x_range = x_range, y_range = y_range, current_full_image = self.full_images[sample_ind]),0,-1) #IM will be shape H x W x Channels ### for the operations involved in register_annotation, We must have the channel dim in the last dimension

		#IF SCALING, SCALE THE IMAGE! 
		#draw the mask 
		M = np.zeros(shape = IM.shape[:2], dtype=bool)
		
		#if verts is less than 0 set to zero.
		verts[verts<0] = 0
		verts[verts[:,0]>M.shape[0] - 1,0] = M.shape[0] - 1
		verts[verts[:,1]>M.shape[1] - 1,1] = M.shape[1] - 1
		skip_inds = [False] + [True if x==0 else False for x in np.sum(np.diff(verts,axis=0),axis=1)]
		verts = verts[~np.array(skip_inds),:]

		rr, cc = skimage.draw.polygon(verts[:,0], verts[:,1], M.shape)
		M[rr,cc] = 1

		""" 
		It is possible that if one does a random crop that a large background may be included. In such a case, we would like to redo the crop.
		However, the mask may still be present; large holes in the annotation may be selected.
		"""
		### IF WE ARE DOING CYCLIC IF, WE NEED TO MAKE SURE THIS PART ONLY FOCUSES ON THE H&E PORTION, WHICH OUGHT TO BE THE FIRST 3 CHANNELS ONLY.
		

		if M.sum()==0:
			print("PAUSE")
			import pdb;pdb.set_trace()

		if mask_out:
			if len(M.shape)!=len(IM.shape): #if IM is multichanneled
				IM = np.where(M[..., None], IM, 0)
			else:
				IM = np.where(M, IM, 0)

		return M, IM

	def crop_annotation(self, IM, M, crop_style = 'center'):
		#create bounding box again.
		inds = np.argwhere(M)
		bbox = [int(np.min(inds[:,0])),
			int(np.max(inds[:,0])),
			int(np.min(inds[:,1])),
			int(np.max(inds[:,1])),
			]
		
		#make sure it is the correct shape for config.INPUT_SPATIAL_SIZE
		bbox_width = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
		centers = [bbox[0] + (bbox_width[0]/2), bbox[2] + (bbox_width[1]/2)]
		#if dimension is smaller than the config.INPUT_SPATIAL_SIZE, for mask, center pad with zeros, for image, center pad with mean of image.
		#if it larger, center crop it.
		#import pdb;pdb.set_trace()
		if crop_style=='center_mask':
			crop_box = [
				int(np.floor(centers[0] - (self.config.INPUT_SPATIAL_SIZE/2))),
				int(np.floor(centers[0] - (self.config.INPUT_SPATIAL_SIZE/2))) + self.config.INPUT_SPATIAL_SIZE,
				int(np.floor(centers[1] - (self.config.INPUT_SPATIAL_SIZE/2))),
				int(np.floor(centers[1] - (self.config.INPUT_SPATIAL_SIZE/2))) + self.config.INPUT_SPATIAL_SIZE,
			]
			scaling = 1.
		
		elif crop_style == 'center':
			crop_box = [
				int(np.floor((IM.shape[0]/2) - (self.config.INPUT_SPATIAL_SIZE/2))),
				int(np.floor((IM.shape[0]/2) - (self.config.INPUT_SPATIAL_SIZE/2))) + self.config.INPUT_SPATIAL_SIZE,
				int(np.floor((IM.shape[1]/2) - (self.config.INPUT_SPATIAL_SIZE/2))),
				int(np.floor((IM.shape[1]/2) - (self.config.INPUT_SPATIAL_SIZE/2))) + self.config.INPUT_SPATIAL_SIZE,
			]
			scaling = 1.

		elif crop_style=='random':
			#pick any indeces within M that have the column index 
			goodrows = np.argwhere((inds[:,0]<IM.shape[0] - self.config.INPUT_SPATIAL_SIZE//2 - 1) & \
					(inds[:,0] - self.config.INPUT_SPATIAL_SIZE//2 >= 0) & \
					(inds[:,1]<IM.shape[1] - self.config.INPUT_SPATIAL_SIZE//2 - 1) & \
					(inds[:,1] - self.config.INPUT_SPATIAL_SIZE//2 >= 0)).flatten()
			
			picked_ind = np.random.choice(goodrows)
			xstart = inds[picked_ind,0] - self.config.INPUT_SPATIAL_SIZE//2#np.random.randint(bbox[0], min([maxx, bbox[1]]))#bbox[1])
			#xstart = xstart if xstart + self.config.INPUT_CHANNEL_SIZE < IM.shape[0] else IM.shape[0] - self.config.INPUT_CHANNEL_SIZE - 1
			ystart = inds[picked_ind,1] - self.config.INPUT_SPATIAL_SIZE//2#np.random.randint(bbox[2], min([maxy, bbox[3]]))#bbox[3])
			#ystart = ystart if ystart + self.config.INPUT_CHANNEL_SIZE < IM.shape[1] else IM.shape[1] - self.config.INPUT_CHANNEL_SIZE - 1
			crop_box = [
				xstart,
				xstart + self.config.INPUT_SPATIAL_SIZE,
				ystart,
				ystart + self.config.INPUT_SPATIAL_SIZE,
			]
			scaling = 1.

		elif crop_style=='scale':
			#take the image, scale it by the largest dimension.
			#whichever has the biggest dimension, keep it, but the other needs to be the same size taken from the centroid center.
			max_shape = np.max(bbox_width) 
			if bbox_width[0]==max_shape: #that means use 
				crop_box = [bbox[0], bbox[1], int(centers[1] - (bbox_width[0]/2)), int(centers[1] - (bbox_width[0]/2)) + bbox_width[0]]
			else:
				crop_box = [int(centers[0] - (bbox_width[1]/2)), int(centers[0] - (bbox_width[1]/2)) + bbox_width[1], bbox[2], bbox[3]]
				#bbox #no, need to make it square, using 
			#sometimes, due to the size of the bounding box, it is possible to get negative values 
			if crop_box[0]<0:
				crop_box[1] = crop_box[1] - crop_box[0]
				crop_box[0] = 0
			if crop_box[2]<0:
				crop_box[3] = crop_box[3] - crop_box[2]
				crop_box[2] = 0
			scaling = max_shape / self.config.INPUT_SPATIAL_SIZE# / max_shape #always take the min.
			#transform the image via scaling
		else:
			print("ERROR!")
			import pdb;pdb.set_trace()

		M = M[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
		IM = IM[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
		#scale the images, if appropriate
		if scaling!=1.:
			tmatrix = np.array([[scaling, 0., 0.],
								[0., scaling, 0.],
								[0., 0., 1.],
			])
			
			max_shape = int(np.max(np.array(bbox_width)*(1./scaling)))
			out_shape = (self.config.INPUT_SPATIAL_SIZE, self.config.INPUT_SPATIAL_SIZE, IM.shape[2])#(max_shape, max_shape, IM.shape[2])
			try:
				IM = ndinterp.affine_transform(IM, tmatrix, order=1,
									output_shape = out_shape)
			except:
				print("ERROR WITH TMATRIX!")
				import pdb;pdb.set_trace()
			M = ndinterp.affine_transform(M, tmatrix[:2,:2], order=0,
								output_shape = out_shape[:2])

			#reapply the thresholds, due to interpolation of affine_transformation.
			M[M<0.5] = 0
			M[M>=0.5] = 1 #binarize and fill holes? Sometimes the scaling really messes things up.
			M = M.astype(bool) #convert back to boolean

		# for dim in range(len(bbox_width)):
		# 	# if bbox_width[dim] < self.config.INPUT_SPATIAL_SIZE:
		# 	# 	#we shouldn't have the problem where the indeces will be SMALLER than the spatial dim.
		# 	if dim==0:
		# 		bbox[0] = int(np.floor(centers[dim] - (self.config.INPUT_SPATIAL_SIZE/2)))
		# 		bbox[1] = bbox[0] + self.config.INPUT_SPATIAL_SIZE
		# 	elif dim==1:
		# 		bbox[2] = int(np.floor(centers[dim] - (self.config.INPUT_SPATIAL_SIZE/2)))
		# 		bbox[3] = bbox[2] + self.config.INPUT_SPATIAL_SIZE

		return IM, M, scaling
	
	def register_annotation(self, M, IM, by_com = False):
		pca = PCA(n_components=2)
		xs = np.repeat(np.arange(M.shape[0])[:,np.newaxis],M.shape[1],axis=1).flatten().squeeze()
		ys = np.repeat(np.arange(M.shape[1])[np.newaxis,:],M.shape[0],axis=0).flatten().squeeze()
		zs = M.flatten().squeeze()
		xs = xs[zs==True]
		ys = ys[zs==True]
		zs = zs[zs==True]

		#get center of mass.
		#mean of xs, mean of ys.
		com = (np.mean(xs),np.mean(ys)) #bear in mind that imshow flips the axes, so the center of mass should be ys, xs.

		# pca.fit(np.stack((ys,xs),axis=1))
		#pca.fit(np.stack((ys - com[1],xs - com[0]),axis=1))
		pca.fit(np.stack((xs,ys),axis=1))
		comp = pca.components_

		#what is the rotation angle between (1,0) and comp[0,:]?
		theta = - np.arctan2(comp[0,1],comp[0,0]) 

		# M = rotate(M,theta*180/np.pi,resize=True)
		# IM = rotate(IM,theta*180/np.pi,resize=True)

		#angle = theta*180/np.piter	

		transform = np.array([[np.cos(theta),-np.sin(theta), 0],
							  [np.sin(theta),np.cos(theta), 0],
							  [0,0,1]])

		### This transform will rotate a matrix COUNTERCLOCKWISE by an angle theta. Note that the angle theta is currently from the (1,0) vector, with postive to the right.

		out_shape = (
						int(np.ceil(np.abs(np.cos(theta)*IM.shape[0]) + np.abs(np.sin(theta)*IM.shape[1]))),
						int(np.ceil(np.abs(np.sin(theta)*IM.shape[0]) + np.abs(np.cos(theta)*IM.shape[1]))),
						IM.shape[2],
					)
		#### This works BECAUSE theta is bound between pi and -pi, according to arctan2.
		
		#import matplotlib.pyplot as plt
		#fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
		#ax1.imshow(IM)
		#We may need to pad the image from center to make this work, at this point. get it to the same size as out shape. Yes, yes definitely.
		change_size = [out_shape[0] - IM.shape[0], out_shape[1] - IM.shape[1]]
		add_size = [max(0, x) for x in change_size]
		#if negative, need to remove these rows after the rotation (I think), if positive need to pad the image first.

		IM = da.pad(IM, ((add_size[0]//2, (add_size[0]//2) + (add_size[0]%2)), 
					(add_size[1]//2, (add_size[1]//2) + (add_size[1]%2)), 
					(0,0),
				   ),
			  )
		
		# ax2.imshow(M)
		"""
		The problem is the output shape is being estimated incorrectly, somehow. If you look at after_rotate.pdf, that's pretty evident.
		"""
		
		# center=0.5*np.array(IM.shape[:2])
		# transform=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
		# offset=-(center-center.dot(transform[:2,:2])).dot(np.linalg.inv(transform[:2,:2]))

		#maybe need to put this back in and remove the center rot offset above
		c_in = 0.5*np.array(IM.shape[:2])
		c_out = 0.5*np.array(IM.shape[:2])
		offset = c_in-c_out.dot(transform[:2,:2])
		# offset_B = c_in-c_out.dot(transform_B[:2,:2])

		IM = ndinterp.affine_transform(IM, transform.T, order=1,
								offset=(offset[0], offset[1], 0),
								output_shape = out_shape)
		#ax3.imshow(IM)
		# IM_B = ndinterp.affine_transform(IM, transform_B, order=1,
		# 						offset=(offset_B[0], offset_B[1], 0),
		# 						output_shape = out_shape)
		#,
								#output_shape=out_shape)
		#(height+int(np.floor(width/(1/y_shear))), width+int(np.floor(height/(1/x_shear))), colors))
		
		IM[IM<0] = 0. #the interpolation will sometimes force negative values.
		IM[IM>1] = 1. #and the otherway, to be safe.

		#convert mask to float, so the interpolation works.

		M = da.pad(M.astype('float32'), ((add_size[0]//2, (add_size[0]//2) + (add_size[0]%2)), 
					(add_size[1]//2, (add_size[1]//2) + (add_size[1]%2)),
				   ),
			  )
		

		M = ndinterp.affine_transform(M, transform[:2,:2].T, order = 1,
									offset=(offset[0], offset[1]),
									output_shape = out_shape[:2])#,
									#output_shape=(height+int(np.floor(width/(1/y_shear))), width+int(np.floor(height/(1/x_shear)))))
		
		#reapply the thresholds, due to interpolation of affine_transformation.
		M[M<0.5] = 0
		M[M>=0.5] = 1
		M = M.astype(bool) #convert back to boolean

		# ax4.imshow(M)
		# plt.savefig("test_rotation.pdf",format='pdf')

		# #determine if the second axis points up or down. They should consistently point one way or the other.
		angle = np.arctan2(comp[0,1], comp[0,0]) - np.arctan2(comp[1,1], comp[1,0])
		# #the pca vectors are always orthogonal. The following if statements make the registration rotation invarient.
		if np.abs(angle) != 3*np.pi/2:
			if angle > 0:
				#flip the y axis. 
				M = da.flipud(M)
				IM = da.flipud(IM)
		else:
			if angle < 0:
				#flip the y axis. 
				M = da.flipud(M)
				IM = da.flipud(IM)


		# #create bounding box again.
		# inds = np.argwhere(M)

		# bbox = [int(np.min(inds[:,0])),
		# 	int(np.max(inds[:,0])),
		# 	int(np.min(inds[:,1])),
		# 	int(np.max(inds[:,1])),
		# 	]
		
		# #make sure it is the correct shape for config.INPUT_SPATIAL_SIZE
		# bbox_width = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
		# centers = [bbox[0] + (bbox_width[0]/2), bbox[2] + (bbox_width[1]/2)]
		# inds = []
		# #if dimension is smaller than the config.INPUT_SPATIAL_SIZE, for mask, center pad with zeros, for image, center pad with mean of image.
		# #if it larger, center crop it.
		# for dim in range(len(bbox_width)):
		# 	if bbox_width[dim] < self.config.INPUT_SPATIAL_SIZE:
		# 		#we shouldn't have the problem where the indeces will be SMALLER than the spatial dim.
		# 		if dim==0:
		# 			bbox[0] = int(np.floor(centers[dim] - (self.config.INPUT_SPATIAL_SIZE/2)))
		# 			bbox[1] = bbox[0] + self.config.INPUT_SPATIAL_SIZE
		# 		elif dim==1:
		# 			bbox[2] = int(np.floor(centers[dim] - (self.config.INPUT_SPATIAL_SIZE/2)))
		# 			bbox[3] = bbox[2] + self.config.INPUT_SPATIAL_SIZE
				
		# M = M[bbox[0]:bbox[1], bbox[2]:bbox[3]]
		# IM = IM[bbox[0]:bbox[1], bbox[2]:bbox[3]]

		if by_com:
			xs = np.repeat(np.arange(M.shape[0])[:,np.newaxis],M.shape[1],axis=1).flatten().squeeze()
			ys = np.repeat(np.arange(M.shape[1])[np.newaxis,:],M.shape[0],axis=0).flatten().squeeze()
			zs = M.flatten().squeeze()
			xs = xs[zs==True]
			ys = ys[zs==True]
			zs = zs[zs==True]
			com = (np.mean(xs),np.mean(ys))

			#center the object now.
			##add rows to the top and left.
			#do dimension 1. 
			####NOW DO THE IMAGE!

			if com[0] < IM.shape[0]/2:
				#add to left. 
				IM = np.concatenate((np.zeros(shape=(int(np.round(IM.shape[0]-2*com[0])),IM.shape[1],IM.shape[2])),IM),axis=0)
			else:
				IM = np.concatenate((IM,np.zeros(shape=(int(np.round(2*com[0] - IM.shape[0])),IM.shape[1],IM.shape[2]))),axis=0)

			if com[1] < IM.shape[1]/2:
				IM = np.concatenate((np.zeros(shape=(IM.shape[0],int(np.round(IM.shape[1]-2*com[1])),IM.shape[2])),IM),axis=1)
			else:
				IM = np.concatenate((IM,np.zeros(shape=(IM.shape[0],int(np.round(2*com[1] - IM.shape[1])),IM.shape[2]))),axis=1)

			### DO THE MASK AS WELL
			if com[0] < M.shape[0]/2:
				#add to left. 
				M = np.concatenate((np.zeros(shape=(int(np.round(M.shape[0]-2*com[0])),M.shape[1]),dtype=bool),M),axis=0)
			else:
				M = np.concatenate((M,np.zeros(shape=(int(np.round(2*com[0] - M.shape[0])),M.shape[1]),dtype=bool)),axis=0)

			if com[1] < M.shape[1]/2:
				M = np.concatenate((np.zeros(shape=(M.shape[0],int(np.round(M.shape[1]-2*com[1]))),dtype=bool),M),axis=1)
			else:
				M = np.concatenate((M,np.zeros(shape=(M.shape[0],int(np.round(2*com[1] - M.shape[1]))),dtype=bool)),axis=1)

			xs = np.repeat(np.arange(M.shape[0])[:,np.newaxis],M.shape[1],axis=1).flatten().squeeze()
			ys = np.repeat(np.arange(M.shape[1])[np.newaxis,:],M.shape[0],axis=0).flatten().squeeze()
			zs = M.flatten().squeeze()
			xs = xs[zs==True]
			ys = ys[zs==True]
			zs = zs[zs==True]
			com = (np.mean(xs),np.mean(ys))
		else:
			xs = np.repeat(np.arange(M.shape[0])[:,np.newaxis],M.shape[1],axis=1).flatten().squeeze()
			ys = np.repeat(np.arange(M.shape[1])[np.newaxis,:],M.shape[0],axis=0).flatten().squeeze()
			zs = M.flatten().squeeze()
			xs = xs[zs==True]
			ys = ys[zs==True]
			zs = zs[zs==True]

			#get center of mass.
			#mean of xs, mean of ys.
			com = (np.mean(xs),np.mean(ys)) #bear in mind that imshow flips the axes, so the center of mass should be ys, xs.


		return IM, M, com
	
	def _get_targets(self):
		all_targets = []
		for sample_ind in range(len(self.sample_file_pairs)):
			# I, anno = self.load_sample(sample_ind=sample_ind)
			# """filter the annotation file to only include certain classes."""
			# #grab the class labels
			# targets = anno['features']['class']
			all_targets = all_targets + self.annotations[sample_ind]['features']['class']
		return all_targets

	def __len__(self):
		if np.any([self.config.dataset_opts['augmentations']['mask_center_crop'], self.config.dataset_opts['augmentations']['scale_annotation']]):
			return np.sum([len(self.annotations[x]['features']['class']) for x in range(len(self.sample_file_pairs))])
		else:
			#some non-descript number, really, we will take random crops 
			return int(1e6)

	# def _reproducible_status(self):
	# 	return self.set_reproducible
	
	# def _set_reproducible(self, state: bool = True):
	# 	self.set_reproducible = state
	
	def __getitem__(self, index):
		"""
		For the validation dataset, in the case where we are doing random augmentations, we need the dataset to be reproducible. The easiest way to achieve that is to get the random seed here.
		However, for the training set, we do not want that. So we need to make this a changable property of the dataset.
		"""

		good_load = False 
		counter = 0
		while not good_load:
			counter += 1  
			if counter > 20: #only during random cropping
				print("Difficulty finding crop that meets requirements...")
				if counter>50:
					assert "", "UNABLE TO FIND CROP, TERMINATING. PLEASE PERFORM A DATA QC."
			if np.any([self.config.dataset_opts['augmentations']['mask_center_crop'], self.config.dataset_opts['augmentations']['scale_annotation']]):
				pinds = self.reind - index  #first, find the sample we should pull
				#as a reminder, self.reind contains the cumsum indexes for each annotated object per image.
				#take the smallest, non-negative number.
				sample_ind = np.argmax(pinds >= 0) #argmax will stop and return the first TRUE index.
				anno_ind =  index - (self.reind[sample_ind-1] + 1 if sample_ind > 0 else 0) # This is just getting the indexing right, linear from 0 -> n for n in N annotations of the Kth sample
			else:
				# if we are doing binary cross entropy, we need to 50% of the time pick a crop from the target sample
				# Below, we randomly pick any annotation from any sample. This will work if we are doing multiclass categorical cross entropy, assuming each sample has only one annotation.
				#if we are doing multiple classes, each class should be equally likely. 
				#if we are doing BCE, we need to overweight the likelihood of picking the target sample.
				#pick any sample, any index. 
				if self.config.train_opts['classifier_loss']=="BCE":
					#weight the target class
					select_file_probs = np.zeros(len(self.sample_file_pairs)) + (0.5 / (len(self.sample_file_pairs) - 1))
					select_file_probs[self.config.train_opts['target_sample_index']] = 0.5
				elif self.config.train_opts['classifier_loss']=="CCE":
					select_file_probs = np.zeros(len(self.sample_file_pairs)) + (1 / (len(self.sample_file_pairs)))

				#import pdb;pdb.set_trace()
				sample_ind = np.random.choice(len(self.annotations), p = select_file_probs)
				#pick an annotation within the sample, using the area of the polygons as the probability of selecting one. 
				anno_ind = np.random.choice(len(self.annotations[sample_ind]['annotation']), p = self.annotations[sample_ind]['features']['probs'])

			#we may want to apply some transformations at this stage, such as random rotations, color distortions, etc.
			M, IM = self.load_instance(sample_ind = sample_ind, object_n = anno_ind, channel = None, mask_out = False) 
			class_label = sample_ind

			IM = IM.astype('float32') / 255. #so far, we don't need to convert away from dask. Torch.Tensor will accept a dask array?
			#M = M.astype('float32') #consists of 1's and zeros.


			#####################################################################
			######################## APPLY AUGMENTATION #########################
			#####################################################################

			##################(1) Apply image registration or not ######################
			
			########### apply rotation, shearing if desired #############

			# imgaug and albumentations could be installed to replace this, but we'll apply numpy operations:
			# I couldn't get this install on my environment to work.
			# rotations, shear, brightness adjustments, noise

			############# ROTATION ################
			# if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['rotation_prob']:
			# 	#what is the rotation angle between (1,0) and comp[0,:]?
			# 	#apply a rotation anywhere from 0-90 degrees.
			# 	theta = np.random.rand()*90 #np.arctan2(comp[0,1],comp[0,0])
			# 	M = rotate(M, theta,resize=True)
			# 	IM = rotate(IM, theta,resize=True)

			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['rotation_prob']:
				#apply left right flipping.
				#make sure these have the right outcome shapes.
				M = da.fliplr(M) #
				IM = da.fliplr(IM)

			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['rotation_prob']:
				M = da.flipud(M) #
				IM = da.flipud(IM)
			
			################ SHEARING ################
			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['shear_prob']:
				x_shear = np.random.rand() * self.config.dataset_opts['augmentations']['shear_mag']
			else:
				x_shear = 0.
			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['shear_prob']:
				y_shear = np.random.rand() * self.config.dataset_opts['augmentations']['shear_mag']
			else:
				y_shear = 0.

			if any([x_shear>0, y_shear>0]):
				height, width, colors = IM.shape
				y_shear = np.random.rand() * self.config.dataset_opts['augmentations']['shear_mag']
				x_shear = np.random.rand() * self.config.dataset_opts['augmentations']['shear_mag']

				transform = np.array([[1, y_shear, 0],
									[x_shear, 1, 0],
									[0, 0, 1]])

				IM = ndinterp.affine_transform(IM, transform,
										offset=(-int(np.floor(width/(1/y_shear))), -int(np.floor(height//(1/x_shear))), 0),
										output_shape=(height+int(np.floor(width/(1/y_shear))), width+int(np.floor(height/(1/x_shear))), colors))
				
				IM[IM<0] = 0. #the interpolation will sometimes force negative values.
				IM[IM>1] = 1. #and the otherway, to be safe.

				M = ndinterp.affine_transform(M.astype('float32'), transform[:2,:2],
											offset=(-int(np.floor(width/(1/y_shear))), -int(np.floor(height//(1/x_shear)))),
											output_shape=(height+int(np.floor(width/(1/y_shear))), width+int(np.floor(height/(1/x_shear)))))
				
				#reapply the thresholds, due to interpolation of affine_transformation.
				M[M<0.5] = 0
				M[M>=0.5] = 1
				M = M.astype(bool) #convert back to boolean

			############ BRIGHTNESS / CONTRAST AUG ################
			#you adjust contrast by doing multiplication, and adjust brightness by adding.
			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['brightcontrast_prob']:
				#choose a brightness value.
				bright_adjust = np.random.rand() * self.config.dataset_opts['augmentations']['brightcontrast_mag']
				#make positive or negative.
				if np.random.rand()>=0.5:
					bright_adjust *= -1 

				IM += bright_adjust

				IM[IM<0] = 0.
				IM[IM>1] = 1.

			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['brightcontrast_prob']:
				#choose a brightness value.
				contrast_adjust = np.random.rand() * self.config.dataset_opts['augmentations']['brightcontrast_mag']
				#make positive or negative.
				if np.random.rand()>=0.5:
					contrast_adjust = 1 - contrast_adjust 
				else:
					contrast_adjust = 1 + contrast_adjust 

				IM *= contrast_adjust
				IM[IM<0] = 0.
				IM[IM>1] = 1.

			################## GAUSSIAN NOISE ######################
			if np.random.rand() >= 1 - self.config.dataset_opts['augmentations']['noise_prob']:
				IM = IM + da.random.normal(0., self.config.dataset_opts['augmentations']['noise_std'], IM.shape).astype('float32')
				#apply clipping.
				IM = da.clip(IM, 0., 1.).astype('float32')

			################## REGISTER IMAGES #####################
			if self.config.dataset_opts['augmentations']['register_annotations']:
				#we may wish to apply shearing and other augmentations before this.
				IM, M, com = self.register_annotation(M, IM, by_com = False) #this converts the image to a numpy array, into memory. If we want to keep it in dask form, you need to not do this

			########################################################
			##################### CROPPING #########################
			########################################################
			if self.config.dataset_opts['augmentations']['mask_center_crop']:
				#(2)
				IM, M, scale = self.crop_annotation(IM, M, crop_style = "center_mask")
				#load the nth object annotation in the .json file, select the color channel index to display, add padding to the edges,
				#and screen the background from the image, if you want.

				M = M.astype('float32') #consists of 1's and zeros.
			
			elif self.config.dataset_opts['augmentations']['random_crop']:
				IM, M, scale = self.crop_annotation(IM, M, crop_style = "center") 
				#find anywhere within the mask to segment.
				M = M.astype('float32') #consists of 1's and zeros.
			
			else:
				#scale the annotation
				#Need to know where the edge of the mask is; first crop the image via the mask, then scale it. 
				IM, M, scale = self.crop_annotation(IM, M, crop_style = "scale")
				#find anywhere within the mask to segment.
				M = M.astype('float32') #consists of 1's and zeros.

			########################################################
			########################################################
			########################################################

			################ REMOVE BACKGROUND ######################
			if self.config.dataset_opts['augmentations']['remove_background']:
				#IM[~M.astype(bool)[:,:,None]] = 0.
				IM = np.where(M.astype(bool)[:, :, None], IM, 0.5) #will be centered at 0 after transform.

			############## IF RANDOM CROPPING, CHECK THE LOADED DATA IS ADEQUATE ###############
			if self.config.dataset_opts['augmentations']['random_crop']:
				#it is possible the loaded image has mostly background. 
				# If the user has selected how much of the image should not contain background, we should check the generated tile and repeat the loading if it doesn't meet the requirement.
				""" CHECK THE IMAGE. """
				bg_pix = np.all(IM[:,:,:3] > (200 / 255.), axis=2).sum() #H&E background signal is gray/white. 
				if bg_pix / (IM.shape[0] * IM.shape[1]) < self.config.dataset_opts['augmentations']['max_background']:
					good_load = True
				else:
					good_load = False
			else:#if we are not doing random cropping, the load is good.
				good_load = True


		#convert classes to a long vector. 
		if self.config.train_opts['classifier_loss']=="CCE":
			target = np.zeros(len(self.sample_file_pairs), dtype=np.float32)
			target[class_label] = 1.
		elif self.config.train_opts['classifier_loss']=="BCE":
			target = np.zeros(2, dtype=np.float32)
			if class_label == self.config.train_opts['target_sample_index']:
				target[1] = 1.
			else:
				target[0] = 1.
		

		#Now, move channel axis back to the 0th dimension (as expected by Pytorch) and stack the Mask in front.
		if isinstance(IM, da.Array):
			if len(IM.shape)>2:
				IM = da.moveaxis(IM, -1, 0) #IM will be shape Channels x H x W 
			else:
				IM = da.expand_dims(IM, axis=0) #add channel dimension
			if self.config.dataset_opts['predict_mask']:
				if not isinstance(M, da.Array):
					M = da.from_array(M)
				IM = da.concatenate([M[None,...], IM], axis=0)
			#convert IM to numpy array, since torch batches cannot accept da.Array types.
			IM = IM.compute()
		
		else:
			if len(IM.shape)>2:
				IM = np.moveaxis(IM, -1, 0) #move the channel to the 1st dimension.
			else:
				IM = np.expand_dims(IM, axis=0) #add channel dimension
			if self.config.dataset_opts['predict_mask']:
				IM = np.concatenate([M[None,...], IM], axis=0)

		#normalize in range -1 - 1
		IM = (IM - 0.5) / 0.5 ##we have a tanh activation in the architecture, so we need this normalization.

		return IM, target #, scale
	
#####################################################################
################### INFERENCE DATASET ###############################
#####################################################################

class GlandDataset_inference(object):
	def __init__(self, config, source:int = 0):

		self.config = config

		image_directory = config.dataset_opts['ome_tif_directory']
		annotation_directory = config.dataset_opts['json_annotation_directory']
		#assuming all the data is contained within a directory with subdirectories named "annotations" and "images"
		#the images should be .ome.tif images, generated from our previous scripts. 
		assert os.path.isdir(image_directory), "Path does not exist ::: {}".format(image_directory)
		assert os.path.isdir(annotation_directory), "Path does not exist ::: {}".format(annotation_directory)

		#create image, annotation filepath pairs
		im_names = find_files_by_pattern(image_directory, [".ome.tif"])

		#for now, exclude any im_names that have 57657 in it, since we are getting permissions sorted.
		#import pdb;pdb.set_trace()
		im_names = [x for x in im_names if "57657.ome.tif" not in x]

		##the number of files in both should be the same.
		#assert len(json_names)==len(im_names), "Found {} files in {}, but only {} in {}".format(len(json_names), annotation_directory, len(im_names), image_directory)

		#now, make the data paired correctly. This may be a user-specific problem, but the easiest solution is the .ome.tif and .json files have the same basename.
		#if this is not the case, you need to write your own code here.

		self.source_file = [os.path.join(image_directory, y) for y in im_names]
		
		#now, check that each file name actually exists. 
		arefiles = np.array([os.path.isfile(x) for x in self.source_file])

		self.source_file = [self.source_file[i] for i,x in enumerate(arefiles) if x]

		import pdb;pdb.set_trace()

		self.source_file = [self.source_file[source]]

		assert len(self.source_file)==1, "Paired file names do not exist! Check code!"
		print("    SOURCE FILE NAME : {}...".format(self.source_file))
		
		#assert np.all(arefiles.flatten()), "Paired file names do not exist! Check code!"

		print("     Loading all samples...")
		self.load_all_samples() #load all the samples and annotations. 
		print("     Complete.")


	def load_da_image(self, ome_file_path):
		#utilize lazy loading using dask, zarr, and xarray.
		current_full_image, channel_names = zarr_read(ome_file_path) #set to True for cyc_IF load, use False for H&E. 
		#if you loaded an H&E, the last dimension contains the channels generally.
		if current_full_image.shape[-1] == 3:
			#assume this is RGB channel. 
			current_full_image = da.moveaxis(current_full_image, -1, 0) #move to zeroth axis.
		
		#given the shape of this image, we need to determine the tile image bounds.
		tile_starts_x = np.arange(0, current_full_image.shape[1], self.config.INPUT_SPATIAL_SIZE)
		tile_ends_x = tile_starts + self.config.INPUT_SPATIAL_SIZE
		if tile_ends_x[-1] > current_full_image.shape[1]:
			tile_ends_x[-1] = current_full_image.shape[1]
		if tile_ends_x[-1] - tile_starts_x[-1] != self.config.INPUT_SPATIAL_SIZE:
			#generally, this will happen on the very last tile, only.
			tile_starts_x[-1] = tile_ends_x[-1] - self.config.INPUT_SPATIAL_SIZE

		tile_starts_y = np.arange(0, current_full_image.shape[2], self.config.INPUT_SPATIAL_SIZE)
		tile_ends_y = tile_starts + self.config.INPUT_SPATIAL_SIZE
		if tile_ends_y[-1] > current_full_image.shape[2]:
			tile_ends_y[-1] = current_full_image.shape[2]
		if tile_ends_y[-1] - tile_starts_y[-1] != self.config.INPUT_SPATIAL_SIZE:
			#generally, this will happen on the very last tile, only.
			tile_starts_y[-1] = tile_ends_y[-1] - self.config.INPUT_SPATIAL_SIZE
		#

		zipped_tiles = [(x, y) for x in zip(tile_starts_x, tile_ends_x) for y in zip(tile_starts_y, tile_ends_y)]

		return current_full_image, zipped_tiles
	
	def load_sample(self, sample_ind: int = 0):
		#self.current_json_file and self.current_ome_file must have been defined. 
		current_ome_file = self.source_file[sample_ind]
		IM, tile_inds = self.load_da_image(current_ome_file)
		return IM, tile_inds
		
	def load_all_samples(self):
		self.full_images = []
		self.all_tile_inds = []
		for sample_ind in range(len(self.source_file)):
			I, tile_inds = self.load_sample(sample_ind=sample_ind)
			self.full_images.append(I)
			self.all_tile_inds.append(tile_inds)

	def load_image_channel(self, x_range: tuple, y_range: tuple, current_full_image: da.Array, channel: int = None):
		#extract the section of the image we want. 
		#after loading the mask, we should also load a subset of coordinates which to pull.
		#use .compute() to load to memory, when needed
		if channel is not None:
			current_im_slice = current_full_image[channel, x_range[0]:x_range[1]-1, y_range[0]:y_range[1]-1]
		else:
			current_im_slice = current_full_image[:, x_range[0]:x_range[1]-1, y_range[0]:y_range[1]-1]
		#self.current_full_image.sel(c=channel, y=slice(int(x_range[0]),int(x_range[1]-1)), x=slice(int(y_range[0]),int(y_range[1]-1)))
		return current_im_slice

	def __len__(self):

		return int(np.sum([len(x) for x in self.all_tile_inds]))


	def __getitem__(self, index):
		"""
		For the validation dataset, in the case where we are doing random augmentations, we need the dataset to be reproducible. The easiest way to achieve that is to get the random seed here.
		However, for the training set, we do not want that. So we need to make this a changable property of the dataset.
		"""
		sample_ind = 0#np.random.choice(len(self.annotations), p = select_file_probs)
		#pick an annotation within the sample, using the area of the polygons as the probability of selecting one. 
		bbox = self.all_tile_inds[sample_ind][index]
		#we may want to apply some transformations at this stage, such as random rotations, color distortions, etc.
		#grab the image. 
		if channel is not None:
			IM = self.load_image_channel(channel = channel, x_range = bbox[0], y_range = bbox[1], current_full_image = self.full_images[sample_ind]) #IM will be H x W
		else:
			IM = da.moveaxis(self.load_image_channel( x_range = bbox[0], y_range = bbox[1], current_full_image = self.full_images[sample_ind]),0,-1) #IM will be shape H x W x Channels ### for the operations involved in register_annotation, We must have the channel dim in the last dimension

		IM = IM.astype('float32') / 255. #so far, we don't need to convert away from dask. Torch.Tensor will accept a dask array?

		#Now, move channel axis back to the 0th dimension (as expected by Pytorch) and stack the Mask in front.
		if isinstance(IM, da.Array):
			if len(IM.shape)>2:
				IM = da.moveaxis(IM, -1, 0) #IM will be shape Channels x H x W 
			else:
				IM = da.expand_dims(IM, axis=0) #add channel dimension
			#convert IM to numpy array, since torch batches cannot accept da.Array types.
			IM = IM.compute()
		
		else:
			if len(IM.shape)>2:
				IM = np.moveaxis(IM, -1, 0) #move the channel to the 1st dimension.
			else:
				IM = np.expand_dims(IM, axis=0) #add channel dimension

		#normalize in range -1 - 1
		IM = (IM - 0.5) / 0.5 ##we have a tanh activation in the architecture, so we need this normalization.

		return IM, bbox

#####################################################################
####################### DATALOADER ##################################
#####################################################################

def seed_worker():
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

def gland_datasets_training(config):
	dataset = GlandDataset(config)
	#dataset._set_reproducible(False)
	train_dataset, test_dataset, _ = split_train_test(dataset, dataset.__len__(), testsize = config.dataset_opts['testsize'], apply_class_balance = config.dataset_opts['class_balance'])
	train_generator = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
	test_generator = DataLoader(test_dataset, config.BATCH_SIZE, shuffle = False, pin_memory=True, drop_last=True) #for reproducibility, shuffle should be set to False.
	#use train_features, train_labels = next(iter(load_traindata)) to get an example
	return train_generator, test_generator

def gland_datasets_training_random(config):
	dataset = GlandDataset(config)
	#dataset._set_reproducible(False)
	train_generator = DataLoader(dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
	#reproducibility is set in the train script by setting the torch and numpy random seeds. The better way to do this would be to make sure the generators are looking at different data, then there is no data leakage problem.
	test_generator = DataLoader(dataset, config.BATCH_SIZE, shuffle = False, pin_memory=True, drop_last=True) #for reproducibility, shuffle should be set to False.
	#use train_features, train_labels = next(iter(load_traindata)) to get an example
	return train_generator, test_generator

def gland_datasets_gradient(config):
	dataset = GlandDataset(config)
	#dataset._set_reproducible(False)
	return dataset

def gland_datasets_latents(config):
	dataset = GlandDataset(config)
	#dataset._set_reproducible(False)
	data_generator = DataLoader(dataset, config.BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
	return dataset.__len__(), dataset.sample_file_pairs, data_generator 

def gland_datasets_inference(config, source: int = 0):
	""" source is the file we want to pull from and map to target """
	dataset = GlandDataset_inference(config)
	#dataset._set_reproducible(False)
	data_generator = DataLoader(dataset, config.BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
	return dataset.__len__(), dataset.sample_file_pairs, dataset.full_images[0].shape, data_generator 

################################################################
################### FASHION MNIST DATALOADER ###################
################################################################

def FashionMNIST_training(config):
	from torchvision import datasets, transforms
	# transform = transforms.Compose([transforms.ToTensor(), 
	# 								transforms.RandomVerticalFlip(p=0.5),
	# 								transforms.RandomRotation(degrees=(0, 180)),
	# 								transforms.Pad((2,2)),
	# 								transforms.Normalize((0.5,),(0.5,),)]) 
	transform = transforms.Compose([transforms.ToTensor(), 
									transforms.Pad((2,2)),
									transforms.Normalize((0.5,),(0.5,),)]) 
	#mean and std have to be sequences (e.g., tuples), therefore add a comma after the values
	#load the data
	train_dataset = datasets.FashionMNIST(root = "/home/groups/CEDAR/eddyc/projects/shared_datasets/Fashion_MNIST", download=False, train=True, transform = transform)
	test_dataset = datasets.FashionMNIST(root = "/home/groups/CEDAR/eddyc/projects/shared_datasets/Fashion_MNIST", download=False, train=False, transform = transform)
	train_generator = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True)
	test_generator = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle = True, pin_memory = True, drop_last = True)
	return train_generator, test_generator



#################################################################
################### UTILITY FUNCTIONS ###########################
#################################################################

def zarr_read(filename):
	"""
	split_channels = boolean, True if you want to display multichanneled images seperately rather than together.
	Use for cycIF or H&E where we don't care to see channel names -- i.e. not split.
	"""
	#grab the channel name metadata
	with tifffile.TiffFile(filename) as tif:
		metadta = ElementTree.fromstring(tif.series[0].pages[0].description)
	metadta = metadta[0][0]
	try:
		channel_names = [(x.attrib)['Name'] for x in metadta if len(x)>0]
	except:
		channel_names = ['Channel {}'.format(x) for x in range(int(float(metadta.attrib['SizeC'])))]

	sample = tifffile.imread(filename, aszarr=True)
	z = zarr.open(sample, mode='r')
	dask_arrays = da.from_zarr(z[int(z.attrs['multiscales'][0]['datasets'][0]['path'])])#.persist()
	return dask_arrays, channel_names
  
def find_files_by_pattern(path, pattern, recursive=True):
	"""
	PURPOSE: Imports ALL files from a chosen folder based on a given pattern
	INPUTS
	-------------------------------------------------------------
	pattern : list of strings with particular patterns, including filetype!
			ex: ["_patched",".csv"] will pull any csv files under filepath with the string "_patched" in its file name.

	filepath : string, path for where to search for files
			ex: "/users/<username>/folder"

	recursive : boolean, True if you wish for the search for files to be recursive under filepath.
	"""
	# generate pattern finding
	fpatterns = ["**{}".format(x) for i,x in enumerate(pattern)]
	if path[-1]!="/":
		path = path + "/"
	all_file_names = set()
	for fpattern in fpatterns:
		file_paths = glob.iglob(path + fpattern, recursive=recursive)
		for file_path in file_paths:
			# skip hidden files
			if file_path[0] == ".":
				continue
			file_name = os.path.basename(file_path)
			all_file_names.add(file_name)

	all_file_names = [file_name for file_name in all_file_names]
	all_file_names.sort()  # sort based on name

	return all_file_names

def save_HE_example(IM, target, save_path):
		assert IM.shape[0]==4, "Example must be H&E, containing mask in the first channel"
		IM = (IM + 1)/2.
		tt = np.argmax(target)
		fig,(ax1,ax2) = plt.subplots(1,2)
		ax1.imshow(IM[0,:,:]) #for force aspect, this is the only way.
		ax1.set_title("Mask")
		ax2.imshow(np.moveaxis(IM[1:,:,:],0,-1))
		ax2.set_title("Class = {}".format(tt))
		plt.savefig(save_path, format='pdf')
		plt.close()


def split_train_test(dataset, data_size, testsize, apply_class_balance=True):

	targets = dataset._get_targets()
	# Stratified Sampling for train and val
	train_idx, validation_idx = train_test_split(np.arange(data_size),
												test_size=testsize,
												random_state=999,
												shuffle=True,
												stratify=targets)
	
	_, counts = np.unique(targets, return_counts=True)
	class_weights = counts / np.sum(counts)
	print("     Class weights = {}".format(class_weights))
	
	if apply_class_balance:
		"""
		We will sample with replacement from the 
		"""
		#class_weights = (1-class_weights) / np.sum(1-class_weights) 
		print("     Applying class balancing...")
		train_weights = np.array([class_weights[y] for y in targets[train_idx]])
		train_weights = (1-train_weights) / np.sum(1 - train_weights)
		val_weights = np.array([class_weights[y] for y in targets[validation_idx]])
		val_weights = (1-val_weights) / np.sum(1 - val_weights)
		#redraw from train_idx and validation_idx to achieve class balance.
		train_idx = np.random.choice(train_idx, size = len(train_idx), replace=True, p = train_weights)
		validation_idx = np.random.choice(validation_idx, size = len(validation_idx), replace=True, p = val_weights)
		print("     Complete.")

	#Subset dataset for train and val	
	train_dataset = Subset(dataset, train_idx)
	validation_dataset = Subset(dataset, validation_idx)

	return train_dataset, validation_dataset, class_weights


###################################################
################### MWE ###########################
###################################################

if __name__ == '__main__':
	#import the configuration file and your arguments.
	import config as configurations
	config = configurations.Config()


	# GD = GlandDataset(config=config)
	# IM,target = GD.__getitem__(1)

	# anno_ind = 1159 #1161 has an irregular shape, and therefore the object is not centered. see /home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Pilot/CE/data/1161_whole_mask.pdf
	# # IM, target, scaling = GD.__getitem__(anno_ind)
	# IM, target = GD.__getitem__(anno_ind)
	# print("Complete!")
	# # #IM will be 4 channeled. 
	# # # for ex in range(6):
	# # # 	print(ex)
	# # # ims = []
	# # N_rc = 10
	# # np.random.seed(1) #make reproducible
	# # import time
	# # import matplotlib.pyplot as plt
	# # start = time.time()
	# # # for jj,anno_ind in enumerate(np.random.choice(np.arange(GD.__len__()), N_rc*N_rc, replace = False)):
	# # # 	IM, target, scaling = GD.__getitem__(anno_ind)
	# # # 	save_HE_example(IM, target, save_path = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Pilot/CE/data/Ex_{}_random.pdf".format(anno_ind))
	# # fig, ax = plt.subplots(N_rc, N_rc)
	# # for i in range(N_rc):
	# # 	for j in range(N_rc):
	# # 		IM, _, _ = GD.__getitem__(anno_ind)
	# # 		IM = (IM+1)/2.
	# # 		ax[i,j].imshow(np.moveaxis(IM[1:,:,:],0,-1))
	# # 		ax[i,j].tick_params(
	# # 						axis='both',       # changes apply to the x-axis
	# # 						which='both',      # both major and minor ticks are affected
	# # 						bottom=False,      # ticks along the bottom edge are off
	# # 						top=False,         # ticks along the top edge are off
	# # 						left=False,
	# # 						right=False,
	# # 						labelbottom=False,
	# # 						labelleft=False) # labels along the bottom edge are off
	# # plt.tight_layout()
	# # plt.savefig("test4.pdf",format="pdf")
	# # endt = time.time()
	# # print("TOTAL TIME for {} examples = {}".format(N_rc*N_rc, endt-start))

	""" USING RANDOM CROPPING? THIS IS YOUR DATA LOADER! """
	if config.dataset_opts['augmentations']['random_crop']:	
		train_gen, val_gen = gland_datasets_training_random(config) #Validation generator is set to be reproducible, but train generator is truly random.
		# """ TEST REPRODUCIBILITY IN VAL GENERATOR """
		# torch.seed()
		# np.random.seed()
		for i,(X,T) in enumerate(train_gen):
			if i==0:
				break 
		# torch.manual_seed(0)
		# np.random.seed(0)
		# for i,(Z,_) in enumerate(val_gen):
		# 	if i==0:
		# 		break 
		# torch.seed()
		# np.random.seed()
		# for i,(X2,_) in enumerate(train_gen):
		# 	if i==0:
		# 		break 
		# torch.manual_seed(0)
		# np.random.seed(0)
		# for i,(Z2,_) in enumerate(val_gen):
		# 	if i==0:
		# 		break 
		# all_targets = np.zeros(config.BATCH_SIZE * 10)
		# for i,(_,T) in enumerate(train_gen):
		# 	print(i+1)
		# 	if i==10:
		# 		break
		# 	else:
		# 		all_targets[int(i*config.BATCH_SIZE):int((i+1)*config.BATCH_SIZE)] = T.numpy().argmax(axis=1)
		
		


		#Note that Z2 and Z should be equal, and X and X2 should NOT be equal. Use torch.all to test!
		
	else:
		train_gen, val_gen = gland_datasets_training(config)