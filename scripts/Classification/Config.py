"""
Purpose:
Base configuration class; 
"""

import numpy as np


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
	BATCH_SIZE=20

	#Number of training epochs
	EPOCHS = 50

	# Number of classification classes
	NUM_CLASSES = 10  

	# Number of input features
	NUM_INPUT_FEATURES = 160

	# Learning rate and momentum
	# In CellPose documentation, these parameters follow
	LEARNING_RATE = 2e-2
	#LEARNING_MOMENTUM = 0.9

	# Weight decay regularization
	WEIGHT_DECAY = 1e-5

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
