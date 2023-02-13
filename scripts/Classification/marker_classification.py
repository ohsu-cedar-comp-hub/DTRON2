"""
Author: Christopher Z. Eddy
Date: 02/06/23
Purpose:
Main script to bring other scripts together to run classification on Cyclic-IF data marker expression data.
"""
import argparse
import os
import sys
import copy
import datetime
import numpy as np
import torch
#import python files
import Model as modellib
import Train as trainlib
#import infer as inferlib
import Config as configurations
import Datasets as datasetlib
import Utils as utilslib

class Marker_Net(object):
	def __init__(self, mode, dataset_path, weights=None, logs=None, subset=None):
		"""
		mode:
			Command for Network
			Options: "'training' or 'inference'
		dataset_path:
			Root directory of the dataset. Training MUST have form:
				dataset_directory
				----/train
				--------/*.csv
				----/val
				--------/*.csv
				Inference MUST have form:
				dataset_directory
				----/*.csv
			The subdirectories must only contain A SINGLE .CSV file. In the future we could group them together. 
		weights:
			Path to weights .tar file if you wish to load previously trained weights.
		logs:
			Logs and checkpoints directory (default=logs/)
		EX:
		CP = Marker_Net(mode="training", dataset_path="../datasets/cell")

		CP.create_model(network_params)
		CP.load_dataset()
		"""
		assert mode in ["training", "inference"], "mode argument must be one of 'training', or 'inference',."
		if dataset_path[-1]=="/":
			dataset_path=dataset_path[:-1]

		print("Dataset directory: ", dataset_path)
		self.dataset_dir = dataset_path

		ROOT_DIR = os.path.abspath("./")
		DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

		if logs:
			self.logs = logs
		else:
			self.logs = DEFAULT_LOGS_DIR
		print("Logs directory: ", self.logs)

		ROOT_DIR = os.path.abspath("./")
		RESULTS_DIR = os.path.join(ROOT_DIR, "results")

		self.submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
		self.submit_dir = os.path.join(RESULTS_DIR, self.submit_dir)
		os.makedirs(self.submit_dir)

		#Create appropriate configuration and model.
		self.config = configurations.Config()#dict(train = configurations.Config())

		#Load weights, if passed.
		self.weights = weights
		if self.weights is not None:
			print("Weights path: ", weights)

		if self.weights is not None:
			#if weights were passed, then load the configuration file that was associated with it.
			self.config = utilslib.load_config_file(os.path.join(os.path.dirname(self.weights),"README.txt"), self.config)

		#display the current configuration settings
		self.config.display()

		self.mode = mode

	def load_dataset(self):
		"""
		Load datasets
		"""
		if self.mode=="training":
			""" We should save where the training/val data was from, and we should also be able to load that path, then verify the presence of the files. """
			#load training and validation.
			#Add the current dataset to the configuration file.
			##################################################
			if not hasattr(self.config,"train_dir"):
				fnames = utilslib.import_filenames(os.path.join(self.dataset_dir,"train"), [".csv"])
				if len(fnames)==0:
					print("Warning: No training *.csv file was found in {}".format(os.path.join(dataset_path,"train")))
					print("No dataset loaded for train!")
					return
				elif len(fnames)>1:
					print("Warning: Multiple *.csv files were found in {}".format(os.path.join(dataset_path,"train")))
					print("No dataset loaded for train!")
					return
				else:
					self.config.train_dir = fnames[0]
			#make sure the file exists in the directory.
			assert os.path.exists(self.config.train_dir), "The designated train .csv file {} in config.train_dir does not exist.".format(self.config.train_dir)
			#del my_dict['key']

			if not hasattr(self.config,"val_dir"):
				fnames = utilslib.import_filenames(os.path.join(self.dataset_dir,"val"), [".csv"])
				if len(fnames)==0:
					print("Warning: No training *.csv file was found in {}".format(os.path.join(self.dataset_dir,"val")))
					print("No dataset loaded for val!")
					return
				elif len(fnames)>1:
					print("Warning: Multiple *.csv files were found in {}".format(os.path.join(self.dataset_dir,"val")))
					print("No dataset loaded for val!")
					return
				else:
					self.config.val_dir = fnames[0]
			#make sure the file exists in the directory.
			assert os.path.exists(self.config.val_dir), "The designated val .csv file {} in config.DATASET_DIR does not exist.".format(self.config.val_dir)
			#del my_dict['key']
				
			if hasattr(self.config,"val_dir") and hasattr(self.config,"train_dir"):
				##################################################
				#we will want to switch 161 and 162. 
				bad_cols = [160,162] #sample name, manually identified classification 
				target_col = 161 #Leiden cluster
				print("Loading training set...")
				self.training_set = datasetlib.cycIF_Dataset(self.config.train_dir, bad_cols=bad_cols, target_col=target_col)
				print("Complete.")
				print("Loading validation set...")
				self.validation_set = datasetlib.cycIF_Dataset(self.config.val_dir, bad_cols=bad_cols, target_col=target_col)
				print("Complete.")
		else:
			##################################################
			if "DATASET_DIR" not in self.config:
				fnames = utilslib.import_filenames(dataset_path, [".csv"])
				if len(fnames)==0:
					print("Warning: No training *.csv file was found in {}".format(dataset_path))
					print("No dataset loaded for train!")
					return
				elif len(fnames)>1:
					print("Warning: Multiple *.csv files were found in {}".format(dataset_path))
					print("No dataset loaded for train!")
					return
				else:
					self.config.DATASET_DIR = fnames[0]
			#make sure the file exists in the directory.
			assert os.path.exists(self.config.DATASET_DIR), "The designated train .csv file {} in config.DATASET_DIR does not exist. ".format(self.config.DATASET_DIR)
			#del my_dict['key']
			#
			bad_cols = [160,161,162]
			print("Loading test set...")
			self.test_set = dataset.cycIF_Dataset(self.config.DATASET_DIR, bad_cols = bad_cols)
			print("Complete.")


	def create_model(self, network_params=None):
		print("Creating model...")
		if self.mode=="training":
			"""
			Note that network_params must be an object with attributes
			class Object(object):
				pass
			network_params=Object()
			network_params.NUM_LAYER_FEATURES = [163,1024,512,256,6]
			network_params.LEARNING_RATE = 0.02
			network_params.WEIGHT_DECAY =  1e-5
			"""
			if network_params is None:
				network_params = self.config
			#build full network.
			self.model = modellib.Generic_MLP(network_params.NUM_LAYER_FEATURES)
			#initialize weights 
			print("initializing weights using uniform xaiver...")
			self.model.apply(modellib.init_weights)
			#build optimizers
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_params.LEARNING_RATE, weight_decay=network_params.WEIGHT_DECAY)
			#Remember that you must call model.eval() or t to set dropout and batch normalization layers to evaluation mode before running inference.
			self.model.train()

		else: #inference or visualize
			#load generator only
			self.model = modellib.Generic_MLP(network_params['NUM_LAYER_FEATURES'])
			self.model.eval()
		print("Complete.")


	def load_weights(self, weights_path, device):
		print("Loading weights....")
		"""
		Load weights from previously saved model
		Check out Train.py for 'writing checkpoint'
		"""
		checkpoint = torch.load(weights_path, map_location=torch.device(device))
		if self.mode=="training":
			if not hasattr(self, "model"):
				print("The 'model' has not yet been loaded. Run 'create_model' first with 'training' mode.")
				return
			self.model.load_model(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.model.train()
		else:
			if not hasattr(self, "model"):
				print("the 'model' has not yet been loaded. Run 'create_model' first.")
				return
			self.model.load_model(checkpoint['model_state_dict'])
			self.model.eval()
		print("Complete.")

	def run_train(self, device):
		#check attributes exist
		if self.mode!="training":
			print("This is a simple verification: Currently, the build mode is not set to 'training'. If you wish to continue, please set the build mode correctly.")
			return
		if not hasattr(self,"training_set"):
			print("Training and validation sets do not yet exist. You must first run 'load_dataset'.")
			return
		if not hasattr(self, "model"):
			print("The 'model' has not yet been created. You must first run 'create_model'.")
			return
		# Parameters
		train_params = {'batch_size': self.config.BATCH_SIZE,
				  'shuffle': True,
				  'num_workers': 4,
				  'drop_last': True,
				  'pin_memory': True} #MAY NEED TO SET NUM_WORKERS TO 0, USE FOR parallelization

		print("Writing configuration to results directory...")
		self.config.write_to_txt(self.submit_dir)

		trainlib.train(train_params, self.training_set, self.validation_set,
					self.model, self.optimizer,
					device, self.submit_dir, self.submit_dir, self.config.NUM_CLASSES,
					max_epochs=self.config.EPOCHS)