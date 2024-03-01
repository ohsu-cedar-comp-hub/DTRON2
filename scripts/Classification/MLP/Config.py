"""
Purpose:
Base configuration class; 
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
	NAME = "MarkerNet"  

	# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
	GPU_COUNT = 1

	# Number of examples to train with on each GPU.
	# Adjust based on your GPU memory and feature sizes. Use the highest
	# number that your GPU can handle for best performance.
	BATCH_SIZE = 500

	#Number of training epochs
	EPOCHS = 100

	#Number of training steps per epoch
	STEPS_PER_EPOCH = 200
	
	#Number of validation steps per epoch to take
	VALIDATION_STEPS = 100

	# Learning rate and momentum
	# In CellPose documentation, these parameters follow
	LEARNING_RATE = 2e-2
	#LEARNING_MOMENTUM = 0.9

	# Weight decay regularization
	WEIGHT_DECAY = 1e-5

	#Gradient Clipping threshold
	GRADIENT_CLIP = 5.

	""" The following should be specific for your dataset """
	dataset_opts = {
			'train_dir' : "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/masterdf_allcells_ACED_cols.csv",#"/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/masterdf_allcells.csv", #"/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/masterdf_allcells.csv",
			'val_dir' : None, #"/home/groups/CEDAR/eddyc/projects/cyc_IF/UMAP/data/Normalized Mean Intensities_processed_robust_CE_20NN_globalthresh_celltyped.csv",
			'test_dir': None,
			'bad_cols': list(np.arange(26, 37)) + [38, 39, 40, 41, 42, 43],#[27,28], #list(range(12)) +  list(range(172,183)),#[x for x in range(183) if x not in [19,  27,  32,  39,  55,  59,  67,  71,  75,  87,  89, 103, 107, 119, 147, 151, 152,  23, 157, 141]], # #MANUALLY IDENTIFIED COLUMNS TO EXCLUDE IN CLASSIFIER! sample name, manually identified classification, CAN BE NONE TYPE
			'target_col': 37,#26, #176,#162, #MANUALLY IDENTIFIED COLUMN OF CSV CONTAINING THE CLASSIFICATION!

			#Run dataset.get_all_targets() to obtain all entries of the target column. Using np.unique, create a unique target mapping list and to get the counts!

			'target_map' : {x:i for i,x in enumerate(np.array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
															13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
															26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
															39., 41., 42., 43., 44., 45., 47.], dtype=int))								
							},
			'class_counts': {i:x for i,x in enumerate(np.array([50209, 48977, 47267, 46304, 33797, 31330, 23252, 22652, 22063,
															21767, 20026, 19837, 18145, 16927, 16856, 16598, 16357, 10387,
															14178, 13718, 13580, 13362, 12397, 11896, 11859, 11254, 11175,
															10439, 10266,  9461,  8894,  8520,  8104,  1887,  7365,  7109,
															6856,  6151,  5112,  2066,  3186,  2865,  2852,  1627,   377,
															154], dtype=int))},

			# 'target_map' : {x:i for i,x in enumerate(np.array(['AMACR high luminal', 'APC M1 macrophage', 'APC M2 macrophage',
			# 										'APC stromal', 'AR+ Smooth Muscle', 'AR+ aSMA+ and Vim+',
			# 										'AR+ other stromal', 'B cells', 'CD4+ CD3+ t cells',
			# 										'CD4- CD3+ t cells', 'CD90+ and Vim+ MSC', 'CD90+ and aSMA+ MSC',
			# 										'EMT/infiltrating vim+', 'Granulocytes', 'M1 macrophage',
			# 										'Mast cells', 'NK cells', 'Smooth Muscle', 'Treg cells', 'basal',
			# 										'luminal', 'mesenchymal/endothelial', 'nerve', 'neuroendocrine',
			# 										'non-APC M2 macrophage', 'other APC immune cells', 'other stromal'],
			# 										dtype='object'))								
			# 				},

			# 'class_counts': {i:x for i,x in enumerate(np.array([ 45306,   2640,   5836,  12745,   8254,   4664,  15863,   1931,
			# 												12702,   7513,   8565,   8300,  20650,   2374,   3106,   9483,
			# 												678,  81255,   1380,  47961, 320395,  32706,   5560,   1390,
			# 												2687,   3908,  31609], dtype=int))
			# 				},
						
		}
	
	train_opts = {
		'classifier_l1_weight': 1e-3,
	}

	# Number of classification classes
	NUM_CLASSES = len(dataset_opts['target_map'].keys())

	#import pdb;pdb.set_trace()
	# Number of input features
	NUM_INPUT_FEATURES = 26#183 - len(np.unique(dataset_opts['bad_cols'] + [dataset_opts['target_col']]))#160

	#Label smoothing during training, to promote uncertainty.
	LABEL_SMOOTHING = 0.1

	# Specify how many features are to be learned in each hidden layer.
	NUM_LAYER_FEATURES = [1024, 512, 256]

	# 
	NUM_LAYER_FEATURES = [NUM_INPUT_FEATURES] + NUM_LAYER_FEATURES + [NUM_CLASSES]

	def __init__(self):
		"""Set values of computed attributes."""
		pass

	def display(self):
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")

	def write_to_txt(self, fpath):
		""" Write configuration values to .txt file """
		with open(os.path.join(fpath, 'README.txt'), 'w') as f:
			f.write("README \n")
			f.write("Training begin date and time: {:%Y%m%dT%H%M%S}".format(datetime.datetime.now()))
			f.write("\n")
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