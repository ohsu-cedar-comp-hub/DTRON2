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
	def __init__(self, mode, config, model_dir = None, log_directory = None, dataset_path=None, weights=None):
		"""
		mode:
			Command for Network
			Options: "'training' or 'inference'
		dataset_path:
			Directory to single csv file ONLY for inference
				----/*.csv
		weights:
			Path to weights .tar file if you wish to load previously trained weights.
		logs:
			Logs and checkpoints directory (default=logs/)
		EX:
		CP = Marker_Net(mode="training", config = config)
		CP = Marker_Net(mode="inference", dataset_path="../datasets/cell.csv")

		CP.create_model(network_params)
		CP.load_dataset()
		"""
		assert mode in ['training', 'inference', 'confusion_matrix']
		if mode in ["inference", "confusion_matrix"]:
			assert weights is not None, "--weights argument must given (path to weights) for generation of data."
		#set the training mode
		self.mode = mode

		self.config = config

		#Load a prior model, if weights argument path is given
		self.weights = weights 
		print("Weights Path: ", weights)
		if self.weights is not None:
			assert os.path.isfile(self.weights), "{} does not exist. Please pass weights path leading to a .tar object.".format(self.weights)
			#load the original configuration file
			self.config = configurations.load_config_file(os.path.join(os.path.dirname(self.weights),"README.txt"), self.config)
			if mode != "train":
				model_dir = os.path.dirname(self.weights)
				log_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.weights))), 'logs', os.path.basename(os.path.dirname(self.weights)))
		# validate the model directory
		if model_dir is None:
			model_dir = os.getcwd()
		self.model_dir = model_dir
		
		#create log directories
		ROOT_DIR = os.path.abspath("./")
		DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
		
		if log_directory is not None:
			self.logs = log_directory
		else:
			self.logs = DEFAULT_LOGS_DIR

		# if mode in ['infer']:
		# 	if self.weights is None:
		# 		RESULTS_DIR = os.path.join(ROOT_DIR, "results")
		# 	else:
		# 		RESULTS_DIR = os.path.dirname(self.weights)
		# 	self.submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
		# 	self.submit_dir = os.path.join(RESULTS_DIR, self.submit_dir)
		# 	os.makedirs(self.submit_dir)
		
		if mode == 'training':
			model_name = "model_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
			self.model_dir = os.path.join(self.model_dir, model_name)
			self.logs = os.path.join(self.logs, model_name)
		
		print("Model directory: ", self.model_dir)
		print("Logs directory: ", self.logs)


		#display the current configuration settings
		# self.config.display()

	def load_dataset(self):
		"""
		Load datasets
		"""
		if self.mode=="training":
			""" We should save where the training/val data was from, and we should also be able to load that path, then verify the presence of the files. """
			#load training and validation.
			#Add the current dataset to the configuration file.
			##################################################
			#make sure the file exists in the directory.
			assert os.path.exists(self.config.dataset_opts['train_dir']), "The designated train .csv file {} in config.dataset_opts['train_dir'] does not exist.".format(self.config.dataset_opts['train_dir'])
			
			if self.config.dataset_opts['val_dir'] is not None:
				assert os.path.exists(self.config.dataset_opts['val_dir']), "The designated val .csv file {} in config.dataset_opts['val_dir'] does not exist.".format(self.config.dataset_opts['val_dir'])
			#del my_dict['key']
			
			##################################################
			if self.config.dataset_opts['val_dir'] is not None:
				print("    Loading training generator...")
				self.training_gen = datasetlib.marker_dataloader(self.config.dataset_opts['train_dir'], num_classes = self.config.NUM_CLASSES, 
							batch_size = self.config.BATCH_SIZE, bad_cols = self.config.dataset_opts['bad_cols'], 
							target_col = self.config.dataset_opts['target_col'], class_mapping = self.config.dataset_opts['target_map'],
							shuffle=True)
				print("    Complete.")
				print("    Loading validation generator...")
				self.validation_gen = datasetlib.marker_dataloader(self.config.dataset_opts['val_dir'], num_classes = self.config.NUM_CLASSES, 
							batch_size = self.config.BATCH_SIZE, bad_cols = self.config.dataset_opts['bad_cols'], 
							target_col = self.config.dataset_opts['target_col'], class_mapping = self.config.dataset_opts['target_map'],
							shuffle=True)
				print("    Complete.")
				
			else:
				print("    Loading training and validation generator...")
				self.training_gen, self.validation_gen = datasetlib.split_marker_dataloader(self.config.dataset_opts['train_dir'], num_classes = self.config.NUM_CLASSES, 
							split_frac = 0.2, batch_size = self.config.BATCH_SIZE, bad_cols = self.config.dataset_opts['bad_cols'], 
							target_col = self.config.dataset_opts['target_col'], class_mapping = self.config.dataset_opts['target_map'],
							shuffle=True)
				print("    Complete.")

		elif self.mode=="confusion_matrix":
			assert os.path.exists(self.config.dataset_opts['train_dir']), "The designated train .csv file {} in config.dataset_opts['train_dir'] does not exist.".format(self.config.dataset_opts['train_dir'])
			self.data_gen = datasetlib.marker_dataloader(self.config.dataset_opts['train_dir'], num_classes = self.config.NUM_CLASSES,
							batch_size = self.config.BATCH_SIZE, bad_cols = self.config.dataset_opts['bad_cols'],
							target_col = self.config.dataset_opts['target_col'], class_mapping = self.config.dataset_opts['target_map'],
							shuffle=False, drop_last = False,
							)

		elif self.mode == "inference":
			assert os.path.exists(self.config.dataset_opts['test_dir']), "The designated train .csv file {} in config.dataset_opts['test_dir'] does not exist.".format(self.config.dataset_opts['test_dir'])
			self.data_gen = datasetlib.marker_dataloader(self.config.dataset_opts['test_dir'], num_classes = self.config.NUM_CLASSES,
							batch_size = self.config.BATCH_SIZE, bad_cols = self.config.dataset_opts['bad_cols'],
							target_col = self.config.dataset_opts['target_col'], class_mapping = self.config.dataset_opts['target_map'],
							shuffle=False, drop_last = False,
							)

		else:
			##################################################
			fnames = utilslib.import_filenames(dataset_path, [".csv"])
			if len(fnames)==0:
				print("    Warning: No training *.csv file was found in {}".format(dataset_path))
				print("    No dataset loaded for train!")
				return
			elif len(fnames)>1:
				print("    Warning: Multiple *.csv files were found in {}".format(dataset_path))
				print("    No dataset loaded for train!")
				return
			#
			print("    Loading test set...")
			self.test_set = dataset.cycIF_Dataset(self.config.DATASET_DIR, bad_cols = self.bad_cols)
			print("    Complete.")


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
			print("    initializing weights using uniform xaiver...")
			self.model.apply(modellib.init_weights)
			#build optimizers
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_params.LEARNING_RATE, weight_decay=network_params.WEIGHT_DECAY)
			#Remember that you must call model.eval() or t to set dropout and batch normalization layers to evaluation mode before running inference.
			self.model.train()

		else: #inference or visualize
			#load generator only
			if network_params is None:
				network_params = self.config

			self.model = modellib.Generic_MLP(network_params.NUM_LAYER_FEATURES)
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_params.LEARNING_RATE, weight_decay=network_params.WEIGHT_DECAY)
			self.model.eval()
		print("    Complete.")
	
	def send_model_to_device(self):
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		#apparently, the models and optimizers are first loaded to cpu. Send to GPU
		print("Sending model to {} device...".format(device))
		self.model.to(device)
		print("Complete.")
		return device

	def optimizer_to(self, device):
		print("Sending optimizer to {} device...".format(device))
		for param in self.optimizer.state.values():
			# Not sure there are any global tensors in the state dict
			if isinstance(param, torch.Tensor):
				param.data = param.data.to(device)
				if param._grad is not None:
					param._grad.data = param._grad.data.to(device)
			elif isinstance(param, dict):
				for subparam in param.values():
					if isinstance(subparam, torch.Tensor):
						subparam.data = subparam.data.to(device)
						if subparam._grad is not None:
							subparam._grad.data = subparam._grad.data.to(device)

	def load_weights(self):
		print("Loading weights....")
		"""
		Load weights from previously saved model
		Check out Train.py for 'writing checkpoint'
		"""
		if self.weights is not None:
			print("Loading model weights...")
			checkpoint = torch.load(self.weights)
			self.model.load_state_dict(checkpoint['model_state_dict']) #let's see if the model tensor pz_params got loaded.
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			epoch = checkpoint['epoch']
			val_ttl_loss = checkpoint['val_ttl_loss']
			print("Complete!")

		else:
			print("self.weights is currently None, so no weights to load. Proceeding with randomly initialized weights and default configuration file")

	def run_train(self):
		#check attributes exist
		if self.mode!="training":
			print("This is a simple verification: Currently, the build mode is not set to 'training'. If you wish to continue, please set the build mode correctly.")
			return
		if not hasattr(self,"training_gen"):
			print("Training and validation generators do not yet exist. You must first run 'load_dataset'.")
			return
		if not hasattr(self, "model"):
			print("The 'model' has not yet been created. You must first run 'create_model'.")
			return
		# # Parameters
		# train_params = {'batch_size': self.config.BATCH_SIZE,
		# 		  'shuffle': True,
		# 		  'num_workers': 1,
		# 		  'drop_last': True,
		# 		  'pin_memory': True} #MAY NEED TO SET NUM_WORKERS TO 0, USE FOR parallelization

		# #if weights were given, load weights.
		# if self.weights is not None:
		# 	self.load_weights()

		device = self.send_model_to_device() #send model to cuda, if possible. 
		self.optimizer_to(device) #send the optimizer to cuda device, if possible

		#make training directory
		os.mkdir(self.model_dir)
		#make logs directory
		os.mkdir(self.logs)

		print("Writing configuration to results directory...")
		#write the configuration to file
		self.config.write_to_txt(self.model_dir)

		trainlib.train(self.config, self.training_gen, self.validation_gen,
					self.model, self.optimizer, device,
					self.logs, self.model_dir)

	def run_inference(self):
		assert hasattr(self, 'data_gen'), "Data generator does not exist. You must first run 'load_dataset'."
		assert hasattr(self, 'model'), "The MLP Model has not yet been created! You must first run 'create_model'."

		import tqdm
		import torch

		N_examples = len(self.data_gen)*self.config.BATCH_SIZE

		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")

		N_batches = int(np.ceil(N_examples / self.config.BATCH_SIZE))

		#apparently, the models and optimizers are first loaded to cpu. Send to GPU
		print("    Sending model to {} device...".format(device))
		self.model.to(device)

		self.model.eval() #set mode to eval to deal with batchnorm from running history

		pred_labels = np.zeros((N_batches * self.config.BATCH_SIZE, self.config.NUM_CLASSES), dtype=int)
		gt_labels = np.zeros(N_batches * self.config.BATCH_SIZE, dtype=int)

		print("    Beginning evaluation...")
		
		bad_labs = 0
		with torch.set_grad_enabled(False):
			with tqdm.tqdm(total = N_batches) as tepoch:
				for ind, X in enumerate(self.data_gen):
					#get batch examples
					start_ind = int(ind * self.config.BATCH_SIZE)
					end_ind = start_ind + X.shape[0] #end ind is non-inclusive it seems.

					if X.shape[0]!=self.config.BATCH_SIZE:
						#zero pad the backend, only necessary for the last batch. 
						X = torch.nn.functional.pad(X, (0,0,0,self.config.BATCH_SIZE - X.shape[0]))

					X = X.to(device) #send X to the appropriate device

					preds = self.model(X) #THESE ARE LOGITS! 
					#we want preds to be a single argmax index.

					if end_ind-start_ind < self.config.BATCH_SIZE:
						preds = preds[:(end_ind-start_ind),...]
						bad_labs += self.config.BATCH_SIZE - (end_ind-start_ind)

					preds = torch.nn.functional.softmax(preds,dim=1) #convert to probabilities

					pred_labels[start_ind:end_ind,:] = preds.cpu().detach().numpy().flatten()

					tepoch.update()
					if ind == (N_batches - 1):
						break #get out of loop.
		
		return pred_labels[:-bad_labs,:] #pred labels is a 2D array, # examples x num classes, where each entry is the class probability. Returning in this format allows the user to report the top N classification.

	def run_eval_dataset(self):
		#Analyze the confusion matrix!
		#Load and evaluate every example within the training set.
		assert hasattr(self, 'data_gen'), "Data generator does not exist. You must first run 'load_dataset'."
		assert hasattr(self, 'model'), "The MLP Model has not yet been created! You must first run 'create_model'."

		import tqdm
		import torch

		N_examples = len(self.data_gen)*self.config.BATCH_SIZE

		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")

		N_batches = int(np.ceil(N_examples / self.config.BATCH_SIZE))

		#apparently, the models and optimizers are first loaded to cpu. Send to GPU
		print("    Sending model to {} device...".format(device))
		self.model.to(device)

		self.model.eval() #set mode to eval to deal with batchnorm from running history

		pred_labels = np.zeros(N_batches * self.config.BATCH_SIZE, dtype=int)
		gt_labels = np.zeros(N_batches * self.config.BATCH_SIZE, dtype=int)

		print("    Beginning evaluation...")
		
		bad_labs = 0
		#import pdb;pdb.set_trace()
		with torch.set_grad_enabled(False):
			with tqdm.tqdm(total = N_batches) as tepoch:
				for ind, (X,Y) in enumerate(self.data_gen):
					#get batch examples
					start_ind = int(ind * self.config.BATCH_SIZE)
					end_ind = start_ind + X.shape[0] #end ind is non-inclusive it seems.
					#Y should be a single, argmax variable.
					gt_labels[start_ind:end_ind] = torch.argmax(Y, dim=1).numpy().flatten().astype(int) #convert back to numpy array, then store

					if X.shape[0]!=self.config.BATCH_SIZE:
						#zero pad the backend, only necessary for the last batch. 
						X = torch.nn.functional.pad(X, (0,0,0,self.config.BATCH_SIZE - X.shape[0]))

					X = X.to(device) #send X to the appropriate device

					preds = self.model(X) #THESE ARE LOGITS! 
					#we want preds to be a single argmax index.

					if end_ind-start_ind < self.config.BATCH_SIZE:
						preds = preds[:(end_ind-start_ind),...]
						bad_labs += self.config.BATCH_SIZE - (end_ind-start_ind)

					preds = torch.argmax(torch.nn.functional.softmax(preds,dim=1),dim=1)

					pred_labels[start_ind:end_ind] = preds.cpu().detach().numpy().flatten()

					tepoch.update()
					if ind == (N_batches - 1):
						break #get out of loop.
		
		return pred_labels[:-bad_labs], gt_labels[:-bad_labs]

	def run_CM(self):
		assert self.mode=="confusion_matrix", "SAFETY WARNING: self.mode must be set to 'confusion_matrix' to run 'run_CM'!"

		print("Predicting classes on given dataset from {}...".format(self.config.dataset_opts['train_dir']))
		preds, gt = self.run_eval_dataset()
		print("Complete!")

		print("Forming the confusion matrix...")
		#form the confusion matrix.
		CM = np.zeros((self.config.NUM_CLASSES, self.config.NUM_CLASSES))
		#let's have targets on the x axis, predictions on the y.

		for row in range(len(preds)):
			CM[int(gt[row]), int(preds[row])] += 1

		print("Complete!")
		
		return CM


####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":

	############################################################
	CP = Marker_Net(mode="training", config=configurations.Config(), 
							 model_dir = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/checkpoints", 
							 log_directory="/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/logs")

	# CP = Marker_Net(mode="confusion_matrix", config=configurations.Config(), 
	# 						 weights = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/checkpoints/model_20231110T100019/MLP_Epoch_91.tar",
	# 						 model_dir = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/checkpoints", 
	# 						 log_directory="/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/logs")

	# CP = Marker_Net(mode="inference", config=configurations.Config(), 
	# 						 weights = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/checkpoints/model_20231110T100019/MLP_Epoch_91.tar",
	# 						 model_dir = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/checkpoints", 
	# 						 log_directory="/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Classification/models/logs")

	############################################################
	CP.create_model()

	print("Loading dataset...")
	CP.load_dataset()

	if CP.weights is not None:
		CP.load_weights()

	if CP.mode == "training":
		CP.run_train()
	
	if CP.mode == "inference":
		print(" NEED TO VALIDATE THIS CODE.")

	if CP.mode == "confusion_matrix":
		CM = CP.run_CM()
		norm = CM / CM.sum(axis=1)[:,None]
		import seaborn as sns 
		import matplotlib.pyplot as plt
		sns.heatmap(norm)
		plt.savefig(os.path.join(CP.model_dir,"confusion_matrix.pdf"),format='pdf')

	if CP.mode == "inference":
		preds = CP.run_inference()
		print("Variable 'preds' currently in memory. Please write to a file if you wish to save.")