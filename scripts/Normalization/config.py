"""
CycIF / H&E Variational Autoencoder Network
Base Configurations Class

Author: Christopher Z. Eddy
Contact: eddyc@ohsu.edu

Done CE 08/07/23
"""

import numpy as np
import os
import datetime


# Base Configuration Class

class Config(object):
	"""Base configuration class. For custom configurations, create a
	sub-class that inherits from this one and override properties
	that need to be changed.
	"""
	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
	# Useful if your code needs to do things differently depending on which
	# experiment is running.
	NAME = "cycIF_VAE"  # Override in sub-classes

	# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
	GPU_COUNT = 1

	# Number of training epochs
	EPOCHS = 100#500 #was 150

	# Batch size
	BATCH_SIZE = 8 #was 16

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	STEPS_PER_EPOCH = 150

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	VALIDATION_STEPS = 50

	# Learning rate and momentum
	LEARNING_RATE = 2e-4 #was 1e-3 #0.02
	LEARNING_MOMENTUM = 0.9

	# Adam optimizer decay terms.
	BETA_1 = 0.5 #default is 0.9
	BETA_2 = 0.999 #default is 0.999

	# Weight decay regularization
	WEIGHT_DECAY = 0.#0.0001

	# Gradient clipping parameter (threshold)
	GRADIENT_CLIP = 10.

	#### DATASET OPTIONS ####
	dataset_opts = {
					# Test size. Train size = 1 - TESTSIZE
					"testsize": 0.2,
					'ome_tif_directory': "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/HandE_annotations/HandE_ometif",#reinhard_ref17633",
					'json_annotation_directory':  '/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/HandE_annotations/tissue_annotations',
					'class_balance': False,
					'train_on_FMNIST': False,
					'predict_mask': False,

					'augmentations': {
						'register_annotations': False, #if annotations should be registered BEFORE Augmentation. If True, no rotation or shearing will be applied.
						'remove_background': False, #True if background (outside annotation) should be set to 0's.
						'mask_center_crop': False, #crop from the center of the annotation
						'random_crop': True, #crop from anywhere within the annotation. Does this in load_instance and then a center crop in crop_annotation
						'max_background': 0.2, #if random_crop is True, this is the maximum percentage of background pixels allowed in the image region for the image to be accepted.
						'scale_annotation': False,
						#if center_crop, random_crop are False, it will scale the annotation to fit within the bounding box region.
						
						'rotation_prob': 0.5,
						'shear_prob': 0.,
						'shear_mag': 0.4, #maximum magnitude of the applied shear in both x and y, but be between 0-1
						'brightcontrast_prob': 0.,
						'brightcontrast_mag': 0.2, #consider this a percent change, must be between 0-1, but recommend small values
						'noise_prob': 0.,
						'noise_std': 0.1**0.5,
					},
					
	}

	#### TRAIN OPTIONS ####
	train_opts = {

		'kld_beta_mag': 0.1, #was 100 # 1000 for 1, 100 for 64, 10 for 600
		'kld_beta_n_cycles': 5,
		'elbo_weight': 1e-3,#1.,#10, #was 0.001
		'class_weight': 1.,#0.1,#5e2,#2000, #was 200
		'discriminator_weight': 10.,
		#'classifier_l1_weight': 5e-1,
		'recon_loss': "distribution", #recon_loss should be MSE or anything else.
		'classifier_loss': 'CCE', #options are "CCE", and "BCE"
		'target_sample_index': 0, #index of the target we want to map all other examples to! Ece has shared 17633 would be good.
		'adversarial_switch_n': 2, #switch adversarial training on this nth epoch.
	}


	# INPUT DATA PARAMETERS
	INPUT_SPATIAL_SIZE = 256
	INPUT_CHANNEL_SIZE = 3 #H&E only, mask channel will be added later if need be.
	NUM_CLASSES = 13 if (not dataset_opts['augmentations']['random_crop']) | (train_opts['classifier_loss']=="CCE") else 2 #Should be equal to the number of samples in your training dataset.

	#### VALIDATE DATASET_OPTS ARGUMENT ####
	if 'scale_annotation' in dataset_opts['augmentations'].keys():
		if dataset_opts['augmentations']['scale_annotation']:
			dataset_opts['augmentations']['random_crop'] = False #must be set to False since we are taking the whole gland.

	if dataset_opts['train_on_FMNIST']:
		INPUT_CHANNEL_SIZE = 1
		dataset_opts['testsize'] = 1/6 #split of Fashion MNIST
		INPUT_SPATIAL_SIZE = 32
		NUM_CLASSES = 10

	if dataset_opts['predict_mask']:
		INPUT_CHANNEL_SIZE += 1

	if train_opts['classifier_loss'] == "BCE":
		NUM_CLASSES = 2

	#### NETWORK OPTIONS ####

	model_opts = { 

		'VAE_IM': {
			'enc_layer_dims': [16, 32, 64, 128, 256], #note, each layer here reduces the spatial dimensionality by half. 
			'dec_layer_dims': [256, 128, 64, 32, 16],
			'latent_dim': 512,
			'kernel_size': 3,
			'pool_size': 2,
			'uppass_style': False,
			'use_AI_model': False,
			'learn_prior': False,
			'discrim_kernel_size': 3,
			'discrim_pool_size': 2,
			'discrim_layer_dims': [16,32,64,128,256]#[8, 16, 32, 64, 128],
		},

		'classifier': {
			'dec_layer_dims': [64, 32, NUM_CLASSES],#[256, 128, 64, 32, NUM_CLASSES],
			'latent_dim': 512,
		},
		  
	}
	
	#Validate the arguments
	assert (train_opts['kld_beta_mag']<=1) and (train_opts['kld_beta_mag']>=0), "Beta magnitude must be less than 1 and greater than or equal to 0."
	#### VALIDATE DATASET_OPTS ARGUMENT ####
	if dataset_opts['train_on_FMNIST']:
		model_opts['VAE_IM']['enc_layer_dims'] = model_opts['VAE_IM']['enc_layer_dims'][1:]
		model_opts['VAE_IM']['dec_layer_dims'] = model_opts['VAE_IM']['dec_layer_dims'][1:]

	assert len(model_opts['VAE_IM']['enc_layer_dims']) == len(model_opts['VAE_IM']['dec_layer_dims']), "Number of encoder/decoder layers must be the same in VAE IM"
	## for some reason, it is claiming "train_opts doesn't exist"
	#assert np.sum([train_opts[key] for key in train_opts.keys()])==1., "Loss weights, specified in config.train_opts must sum to 1 ::: current sum = {}".format(np.sum([train_opts[key] for key in train_opts.keys()]))
	

	def display(self):
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")

	def write_config_txt(self,fpath):
		""" Write configuration values to .txt file """
		with open(os.path.join(fpath,'README.txt'), 'w') as f:
			f.write("README \n")
			f.write("Training begin date and time: {:%Y%m%dT%H%M%S}".format(datetime.datetime.now()))
			f.write("\n")
			f.write("Notes:\n")
			f.write("CONFIGURATION SETTINGS: \n")
			for a in dir(self):
				if not a.startswith("__") and not callable(getattr(self, a)):
					f.write("{:30} {}".format(a, getattr(self, a)))
					f.write("\n")

### LOAD CONFIG SETTINGS IF AVAILABLE.
def load_config_file(fpath, config):
	if not os.path.isfile(fpath):
		print("\n README.txt config file does not exist in weights path. Proceeding with default configuration...\n")
	else:
		import ast
		import re
		README_dict = {}
		with open(fpath) as file:
			lines = [line.rstrip() for line in file]
		begin=False

		for i,line in enumerate(lines):
			if line=='CONFIGURATION SETTINGS:':
				#want to begin on the NEXT line.
				begin=True
				continue

			if begin==1:
				(key, val) = line.split(maxsplit=1)
				try:
					#anything that is not MEANT to be a string.
					#mostly this does fine on its own.
					README_dict[key] = ast.literal_eval(val)
				except:
					try:
						#messes up list attributes where there spaces are not uniform sometimes.
						README_dict[key] = ast.literal_eval(re.sub("\s+", ",", val.strip()))
					except:
						README_dict[key] = val

		print("\n Replacing default config key values with previous model's config file... \n")
		for func in dir(config):
			if not func.startswith("__") and not callable(getattr(config, func)):
				#print("{:30} {}".format(a, getattr(self, a)))
				if func in README_dict.keys():
					#special case if it is a dictionary.
					if isinstance(README_dict[func],dict):
						#change keys if they exist in config.
						#get the dictionary from config.
						config_dict = getattr(config, func)
						#change values.
						#get keys in README_dict[func]
						for RM_key in README_dict[func].keys():
							#if RM_key in config_dict.keys():
							#	#reset value.
							config_dict[RM_key] = README_dict[func][RM_key]
							#else:
							#	#add new key...
						#set into config.
						setattr(config, func, config_dict)
					else:
						setattr(config, func, README_dict[func])

	return config