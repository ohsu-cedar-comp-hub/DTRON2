import os
#import sys
# sys.path.append("..")
#import python files
import model as modellib
import train as trainlib
import config as configurations
import utils as utilslib
#import dataset
import dataset as datasetlib
import datetime
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torchsummary import summary


###################################################################################
###################################################################################
####################  Normalization Network IMPLEMENTATION  #######################
###################################################################################
###################################################################################

class Normalize_Network(object):
	"""
	Encapsulates the TFBS active binding prediction model functionality.
	"""

	def __init__(self, mode, config, model_dir = None, log_directory=None, weights = None):
		"""
		mode = either "training" or "inference"
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights

		“Advice is a dangerous gift, even from the wise to the wise, and all courses may run ill.”
		"""
		assert mode in ['train', 'infer', 'generate', 'reconstruct', 'visualize', 'gradient', 'get_all_latents']
		if mode in ["infer", "reconstruct", "generate", "visualize", "gradient", "get_all_latents"]:
			assert weights, "--weights argument must given (path to weights) for generation of data."
		#set the training mode
		self.mode = mode

		#Create appropriate configuration
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
				log_directory = os.path.join(os.path.dirname(os.path.dirname(self.weights)), 'logs', os.path.basename(os.path.dirname(self.weights)))
		
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

		if mode in ['infer']:
			if self.weights is None:
				RESULTS_DIR = os.path.join(ROOT_DIR, "results")
			else:
				RESULTS_DIR = os.path.dirname(self.weights)
			self.submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
			self.submit_dir = os.path.join(RESULTS_DIR, self.submit_dir)
			os.makedirs(self.submit_dir)
		
		if mode == 'train':
			model_name = "model_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
			self.model_dir = os.path.join(self.model_dir, model_name)
			self.logs = os.path.join(self.logs, model_name)
		
		print("Model directory: ", self.model_dir)
		print("Logs directory: ", self.logs)
		if mode in ['infer']:
			print("Results directory: ", self.submit_dir)
			

	#############################################################################
	#############################################################################
	def create_model(self):
		"""
		Arise, arise, riders of Rohan!
		"""
		print("Building Gland network...")

		self.torch_model = modellib.cycIF_VAE(self.config)
		self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.torch_model.parameters()),
                       lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY, amsgrad=True)
		#torch.optim.SGD(self.torch_model.parameters(), lr=self.config.LEARNING_RATE)
		#torch.optim.Adam(self.torch_model.parameters())#, lr = self.config.LEARNING_RATE) ### Adam should be used with their default parameters.
		### https://stackoverflow.com/questions/55770783/model-learns-with-sgd-but-not-adam
		#torch.optim.AdamW(self.torch_model.parameters()) ### Adam should be used with their default parameters.

		print("Complete.")

	def send_model_to_device(self):
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		#apparently, the models and optimizers are first loaded to cpu. Send to GPU
		print("Sending model to {} device...".format(device))
		self.torch_model.to(device)
		print("Complete.")
		return device
	
	def optimizer_to(self, device):
		print("Sending optimizer to {} device...".format(device))
		for param in self.optim.state.values():
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
	#############################################################################
	#############################################################################

		
	def load_dataset(self, source = None):
		"""
		The beacons are lit! Gondor calls for aid!
		And Rohan will answer. Muster the Rohirrim. Assemble the army at Dunharrow. As many men as can be found.
		"""
		print("Loading Dataset...")
		if self.mode in ['train','reconstruct','visualize']:
			if self.config.dataset_opts['train_on_FMNIST']:
				self.train_gen, self.val_gen = datasetlib.FashionMNIST_training(self.config)
			else:
				if self.config.dataset_opts['augmentations']['random_crop']:
					print("    Preparing random_crop dataset...")
					self.train_gen, self.val_gen = datasetlib.gland_datasets_training_random(self.config)
				else:
					self.train_gen, self.val_gen = datasetlib.gland_datasets_training(self.config)
			print("Complete.")
		elif self.mode == 'infer':
			self.dataset_len, self.dataset_files, self.source_shape, self.infer_gen = datasetlib.gland_datasets_inference(self.config, source = source)
		elif self.mode == 'gradient':
			self.dataset = datasetlib.gland_datasets_gradient(self.config)
		elif self.mode == "get_all_latents":
			self.dataset_len, self.dataset_files, self.data_gen = datasetlib.gland_datasets_latents(self.config)
			
	def load_weights(self):
		
		if self.weights is not None:
			print("Loading model weights...")
			checkpoint = torch.load(self.weights)
			self.torch_model.load_state_dict(checkpoint['model_state_dict']) #let's see if the model tensor pz_params got loaded.
			self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
			epoch = checkpoint['epoch']
			val_ttl_loss = checkpoint['val_ttl_loss']
			print("Complete!")

		else:
			print("self.weights is currently None, so no weights to load. Proceeding with randomly initialized weights and default configuration file")

	####################################################################
	####################################################################
	#################### EXECTUABLE COMMANDS ###########################
	####################################################################
	####################################################################

	def run_store_latent_params(self):
		#validate arguments 
		assert self.mode == "get_all_latents", "SAFETY WARNING: self.mode is set to '{}', but must be set to 'get_all_latents' to run inference.".format(self.mode)
		assert hasattr(self, 'torch_model'), "Model has not yet been initialized. First run 'self.create_model'!"
		assert hasattr(self, 'data_gen'), "data_gen has not yet been initialized. First run 'self.load_dataset'!"

		import tqdm 
		import h5py
		#set N examples so that we evaluate ALL data.
		N_examples = self.dataset_len#len(self.data_gen)*self.config.BATCH_SIZE
		N_batches = int(np.ceil(N_examples / self.config.BATCH_SIZE)) #should be equal to len(self.data_gen)??

		#what happens to the batch dim? well, if we set the model to eval mode, the layers like batchnorm which does running stats will not need to calculate them and will not raise any errors.
		#we'll need to test that.
		mu_preds = np.zeros((N_examples, self.config.model_opts['VAE_IM']['latent_dim']))
		sigma_preds = np.zeros((N_examples, self.config.model_opts['VAE_IM']['latent_dim']))

		print("    Beginning evaluation...")
		
		#bad_labs = 0
		targets = np.zeros(N_examples)
		print("MAKE SURE THE TARGETS VARIABLE WORKS!")
		with torch.set_grad_enabled(False):
			with tqdm.tqdm(total = N_batches) as tepoch:
				for ind, (X, target) in enumerate(self.data_gen):
					#get batch examples
					start_ind = int(ind * self.config.BATCH_SIZE)
					end_ind = start_ind + X.shape[0] #end ind is non-inclusive it seems.
					targets[start_ind:end_ind] = target.numpy().argmax(axis=1)

					# if X.shape[0]!=self.config.BATCH_SIZE:
					# 	#zero pad the backend, only necessary for the last batch. 
					# 	X = torch.nn.functional.pad(X, (0,0,0,0,0,self.config.BATCH_SIZE - X.shape[0]))

					X = X.to(device) #send X to the appropriate device

					mu, sig = self.torch_model.get_posterior_params(X) #THESE are outputs of the encoder portion of the network

					# if end_ind-start_ind < self.config.BATCH_SIZE:
					# 	preds = preds[:(end_ind-start_ind),...]
					# 	bad_labs += self.config.BATCH_SIZE - (end_ind-start_ind)
					
					#these parameters should be 2D, example x params
					mu_preds[start_ind:end_ind,:] = mu.cpu().detach().numpy()
					sigma_preds[start_ind:end_ind,:] = sig.cpu().detach().numpy()

					tepoch.update()

		### WRITE mu_preds and sigma_preds to h5 file. 
		hpath = os.path.join(self.model_dir, "gland_latents.h5")
		loaded = h5py.File(hpath,'w')
		loaded.create_dataset("dataset_files", data=self.dataset_files)
		loaded.create_dataset("labels", data=targets)
		loaded.create_dataset("mu", data=mu_preds)
		loaded.create_dataset("sig", data=sigma_preds)
		loaded.close()
		print("COMPLETE. Wrote file to {}!".format(hpath))


	def run_interpolation(self, device, N_draw = 2, interpolation_size = 16, obj_indeces = None, obj_distances = None):
		"""
		INPUTS:
		obj_indeces = Numpy integer array of size N integers, indeces of selected objects to run inference interpolation between. 
					  We will find the latent spaces of these N examples, and interpolate latent space between them to a total length of 'interpolation_size'

		obj_distances = Numpy float array of length N (same as obj_indeces). Specifies the calculated distances between the latent distributions.

		N_draw = integer, number of latent spaces to draw and decode from the latent space parameters. 

		interpolation_size = integer, total interpolation size.

		"""
		#validate arguments
		assert self.mode == "gradient", "SAFETY WARNING: self.mode is set to '{}', but must be set to 'gradient' to run interpolation.".format(self.mode)
		assert hasattr(self, 'torch_model'), "Model has not yet been initialized. First run 'self.create_model'!"
		if self.config.dataset_opts['train_on_FMNIST']:
			assert hasattr(self, 'train_gen'), "train_gen has not yet been initialized. First run 'self.load_dataset'!"
		else:
			assert hasattr(self, 'dataset'), "Dataset has not yet been initialized. First run 'self.load_dataset'!"
		if obj_indeces is not None:
			assert interpolation_size > len(obj_indeces), "The length of 'obj_indeces' is {}, but the 'interpolation_size' is smaller than the number of examples.".format(len(obj_indeces))
		if obj_indeces is not None:
			if obj_distances is not None:
				assert len(obj_distances) == obj_indeces.shape[0] - 1, "'obj_distances' argument must have the same number of entries as 'obj_indeces'; got shape {}".format(obj_distances.shape)

		if obj_indeces is None:
			print("Grabbing two different examples...")
			""" I think we need to implement this? """
		else:
			print("Grabbing example indices {}...".format(obj_indeces))

		if not self.config.dataset_opts['train_on_FMNIST']:
			data = utilslib.get_gradient_examples(self.dataset, obj_indeces)
		else:
			data = utilslib.get_gradient_examples(self.train_gen)
		data = data.to(device)


		print("Drawing {} example latent space(s) from latent distribution...".format(N_draw))
		og_latents = self.torch_model.get_latents(data, K = N_draw) ### T
		#grab just the first group
		#linearly interpolate between these latents. 
		print("Doing linear interpolation of N = {} between drawn latent spaces, using provided distances to determine interpolation steps...".format(interpolation_size))
		#the provided indices already tell us the ordering. We need to find the distances.
		if obj_indeces is None:
			obj_distances = np.array([1. for _ in range(og_latents.shape[1] - 1)])
		if obj_distances is None:
			obj_distances = np.array([1. for _ in range(og_latents.shape[1] - 1)])
		import pdb;pdb.set_trace()
		#find the minimum distance in the list.
		obj_distances = obj_distances / obj_distances.min()
		#now, divide by the sum to get the percent of the interpolation they should have.
		obj_distances = obj_distances / obj_distances.sum()
		#multiply by the interpolation_size and subtract 1 to determine the number of frames that should be interpolated between each example!
		obj_distances = (np.ceil(obj_distances * interpolation_size)).astype(int)
		#make sure all are greater than or equal to 0.
		obj_distances[obj_distances<2] = 2 #must be a minimum of two when doing the interpolation between two slices!
		#now, do the interpolation between each, stack the results.
		latents = []

		for i,interp_size in enumerate(obj_distances):
			if i<len(obj_distances)-1:
				latents.append(torch.nn.functional.interpolate(og_latents[:,i:i+2,...].unsqueeze(0), size=(interp_size, og_latents.shape[-1]), mode='bilinear').squeeze(0)[:,:-1,...]) #exclude the last point.
			else:
				latents.append(torch.nn.functional.interpolate(og_latents[:,i:,...].unsqueeze(0), size=(interp_size, og_latents.shape[-1]), mode='bilinear').squeeze(0)) #include the last point
		#stack the latents. 
		latents = torch.cat(latents, dim = 1)
		
		interpolation_size = latents.shape[1] #overwrite the interpolation size, since technically it could be different now.

		#reshape latents to run through code.
		latents = latents.view(int(N_draw * latents.shape[1]), latents.shape[-1])
		#reconstruct
		print("Calculating reconstructed outputs from interpolated latents...")
		recon = self.torch_model.reconstruct_from_latents(latents)
		if N_draw > 1:
			recon = recon.view(N_draw, interpolation_size, *recon.shape[1:])
		else:
			recon = recon.unsqueeze(0) #add the dimension via an unsqueeze, since view will not work to expand the dimensions.

		recon = recon.cpu().detach().numpy() #send to CPU
		recon = (recon + 1)/2

		# ## UNCOMMENT TO SAVE VIDEO
		# for i in range(interpolation_size):
		# 	print(i)
		# 	fig,ax = plt.subplots(1, figsize = (3,3), dpi = 200)
		# 	ax.imshow(np.moveaxis(recon[0, i, :min(3,recon.shape[1]),:,:],0,-1))
		# 	ax.tick_params(
		# 		axis='both',       # changes apply to the x-axis
		# 		which='both',      # both major and minor ticks are affected
		# 		bottom=False,      # ticks along the bottom edge are off
		# 		top=False,         # ticks along the top edge are off
		# 		left=False,
		# 		right=False,
		# 		labelbottom=False,
		# 		labelleft=False) # labels along the bottom edge are off
		# 	plt.savefig(os.path.join("/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/results/movie_recon/min_span_tree", "movie_recon_%02d.png"%i),format='png')
		# 	plt.close()
		### UNCOMMENT TO SAVE RECONSTRUCTION!
		print("Producing output image...")
		#save reconstruction.
		fig,ax = plt.subplots(interpolation_size, N_draw, figsize = (int(N_draw), int(interpolation_size)), dpi=200)
		for i in range(interpolation_size):
			if N_draw>1:
				for j in range(N_draw):
					if not self.config.dataset_opts['predict_mask'] or self.config.dataset_opts['train_on_FMNIST']:
						ax[i,j].imshow(np.moveaxis(recon[j, i, :min(3,recon.shape[1]),:,:],0,-1))
					else:
						ax[i,j].imshow(np.moveaxis(recon[j, i, 1:4,:,:],0,-1)) #take just the first 3 color channels
					ax[i,j].tick_params(
							axis='both',       # changes apply to the x-axis
							which='both',      # both major and minor ticks are affected
							bottom=False,      # ticks along the bottom edge are off
							top=False,         # ticks along the top edge are off
							left=False,
							right=False,
							labelbottom=False,
							labelleft=False) # labels along the bottom edge are off
			else: #okay this is annoying, but if we call subplots when one of the arguments is 1, it doesn't do a square indexing.
				if not self.config.dataset_opts['predict_mask'] or self.config.dataset_opts['train_on_FMNIST']:
					ax[i].imshow(np.moveaxis(recon[0, i, :min(3,recon.shape[1]),:,:],0,-1))
				else:
					ax[i].imshow(np.moveaxis(recon[0, i, 1:4,:,:],0,-1)) #take just the first 3 color channels
				ax[i].tick_params(
						axis='both',       # changes apply to the x-axis
						which='both',      # both major and minor ticks are affected
						bottom=False,      # ticks along the bottom edge are off
						top=False,         # ticks along the top edge are off
						left=False,
						right=False,
						labelbottom=False,
						labelleft=False) # labels along the bottom edge are off
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.model_dir, "linear_recon.pdf"),format='pdf')
		plt.close()

		#save the examples
		fig, ax = plt.subplots(data.shape[0], dpi=200)
		data = data.cpu().detach().numpy()
		for ex in range(data.shape[0]):
			ax[ex].imshow(np.moveaxis((data[ex,...]+1)/2 , 0, -1))
			ax[ex].tick_params(
						axis='both',       # changes apply to the x-axis
						which='both',      # both major and minor ticks are affected
						bottom=False,      # ticks along the bottom edge are off
						top=False,         # ticks along the top edge are off
						left=False,
						right=False,
						labelbottom=False,
						labelleft=False) # labels along the bottom edge are off
			
		plt.tight_layout()
		plt.savefig(os.path.join(self.model_dir, "examples.pdf"),format='pdf')
		plt.close()
		print("Complete.")

	def run_inference(self, target:int = 0):
		""" target should be an integer """
		assert target < self.config.NUM_CLASSES, "target variable is {}, which is larger than the model that was trained with config.NUM_CLASSES = {} classes!".format(target, self.config.NUM_CLASSES)
		assert hasattr(self, 'infer_gen'), "Data generator has not yet been initialized. First run 'self.load_dataset'!"
		assert hasattr(self, 'torch_model'), "Model has not yet been initialized. First run 'self.create_model'!"
		assert hasattr(self, 'dataset_files', "Data generator has not yet been initialized correctly, needs variable self.dataset_files")
		assert len(self.dataset_files)==1, "self.dataset_files should only be length 1, for now. Code in the future should be set up to evaluate multiple files, but for now only available for 1 file."
		assert self.mode == "infer", "SAFETY WARNING: self.mode is set to '{}', but must be set to 'infer' to run inference.".format(self.mode)

		import h5py 
		#create a new h5 file:
		source_file = self.dataset_files[0] #check this should only be length 1.
		loaded = h5py.File(os.path.join(self.model_dir, "recon_s_{}_t_{}.hdf5".format(self.source_file, target)),'w')
		indices = loaded.create_group("indices")
		recons = loaded.create_group("reconstructions")
		loaded.create_dataset("Shape", data = self.source_shape)
		## may need to define a new dataset type for this, since the current one only returns labeled data
		self.torch_model.eval()
		with torch.set_grad_enabled(False):
			with tqdm.tqdm(total = self.dataset_len) as tepoch:
				for ind, (X, inds) in enumerate(self.infer_gen):
					#get batch examples
					start_ind = int(ind * self.config.BATCH_SIZE)
					end_ind = start_ind + X.shape[0] #end ind is non-inclusive it seems.
					targets[start_ind:end_ind] = target.numpy().argmax(axis=1)

					if X.shape[0]!=self.config.BATCH_SIZE:
						#zero pad the backend, only necessary for the last batch. 
						try:
							X = torch.nn.functional.pad(X, (0,0,0,0,0,self.config.BATCH_SIZE - X.shape[0]))
						except:
							import pdb;pdb.set_trace()

					X = X.to(device) #send X to the appropriate device

					recon, _ = self.torch_model.reconstruct(X, target) #THESE are outputs of the encoder portion of the network
					
					recon = recon.cpu().detach().numpy()[start_ind:end_ind, ...] #bring to memory.
					
					#store into the h5 file, with a new dataset name
					indices.create_dataset("{}".format(f"{ind:03}"), data = inds.numpy())
					recons.create_dataset("{}".format(f"{ind:03}"), data = recon)

					tepoch.update()

		print("COMPLETE :) :::: Results saved in {}".format(os.path.join(self.model_dir, "recon_s_{}_t_{}.hdf5".format(self.source_file, target))))
		loaded.close()
		

	def run_train(self):
		assert hasattr(self, 'train_gen'), "Data generators have not yet been initialized. First run 'self.load_dataset'!"
		assert hasattr(self, 'torch_model'), "Model and optimizer has not yet been initialized. First run 'self.create_model'!"
		assert self.mode == "train", "SAFETY WARNING: self.mode is set to '{}', but must be set to 'train' to run training.".format(self.mode)

		#if weights were given, load weights.
		if self.weights is not None:
			self.load_weights()
		else:
			print("Initializing model parameters to max range -{} to {}".format(0.08, 0.08))
			# Apply Xavier initialization to all layers of your model
			self.torch_model.apply(utilslib.xavier_init_range_all_layers)

		device = self.send_model_to_device()
		self.optimizer_to(device) #this is a function, defined below.

		self.torch_model.train() #set mode of the model

		#make training directory
		os.mkdir(self.model_dir)
		#make logs directory
		os.mkdir(self.logs)

		#write the configuration to file
		self.config.write_config_txt(self.model_dir)

		trainlib.train(training_generator = self.train_gen, 
		 			   validation_generator = self.val_gen,
					   model = self.torch_model, 
					   optimizer = self.optim,
					   log_path = self.logs,
					   save_path = self.model_dir,
					   device = device,
					   config = self.config)

if __name__ == "__main__":

	# GN = Normalize_Network(mode='train', config=configurations.Config(), 
	# 						 model_dir = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Normalization/models/checkpoints", 
	# 						 log_directory="/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Normalization/models/logs")
	
	# GN = Normalize_Network(mode='train', config=configurations.Config(), 
	# 						 weights = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints/model_20230905T150353/Gland_net_Epoch_99.tar",
	# 						 model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
	# 						 log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")

	# GN = Normalize_Network(mode='generate', config=configurations.Config(), 
	# 	    				 weights = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints/model_20230930T004934/Gland_net_Epoch_92.tar",
	# 						 model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
	# 						 log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")

	GN = Normalize_Network(mode='reconstruct', config=configurations.Config(), 
		    				 weights = "/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/scripts/Normalization/models/checkpoints/model_20231213T123106/Gland_net_Epoch_91.tar",
							 model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
							 log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")

	# GN = Normalize_Network(mode='visualize', config=configurations.Config(), 
	# 						weights = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints/model_20230918T214236/Gland_net_Epoch_99.tar",
	# 						model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
	# 						log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")

	# GN = Normalize_Network(mode='gradient', config=configurations.Config(), 
	# 						weights = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints/model_20231120T130643/Gland_net_Epoch_98.tar",
	# 						model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
	# 						log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")

	# GN = Normalize_Network(mode='get_all_latents', config=configurations.Config(), 
	# 						weights = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints/model_20231120T130643/Gland_net_Epoch_98.tar",
	# 						model_dir = "/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/checkpoints", 
	# 						log_directory="/home/groups/CEDAR/eddyc/projects/Nvidia-CEDAR/CEDAR-NVIDIA-Internal/Model/pseudotime_VAE/models/logs")
	
	####################################################################################
	####################################################################################
	####################################################################################

	if GN.mode == "reconstruct":
		GN.config.BATCH_SIZE = 8 #reset the batch size.

	if GN.mode not in ['generate']:
		#load dataset.
		GN.load_dataset()
		#import pdb;pdb.set_trace()
		
	#create the model
	GN.create_model()
	
	#Execute
	if GN.mode == "train":
		print("running training...")
		GN.run_train()

	elif GN.mode == "generate":
		#see the old model weights: 
		#TF_net.torch_model.stem.stem[0].weight.shape
		GN.load_weights()
		GN.torch_model.eval()
		device = GN.send_model_to_device()

		print("running generation...")
		gen_dta = GN.torch_model.generate(N = 5) #where N is the number of examples to draw
		## need to detach if you want to use.
		## gen_dta = gen_dta.cpu().detach().numpy()
		#to save, just use utilslib.save_generated_images(gen_dta, save_path = "/path/to/save.pdf")
	
	elif GN.mode == "reconstruct":
		GN.load_weights()
		GN.torch_model.eval()
		device = GN.send_model_to_device()
		GN.config.BATCH_SIZE = 8
		print("running reconstruction...")
		#grab a data sample.
		data,source = next(iter(GN.train_gen))
		target = torch.zeros(GN.config.BATCH_SIZE, GN.config.NUM_CLASSES)
		target[:,0] = 1.
		recon_data, latents = GN.torch_model.reconstruct(data = data.to(device),target = target.to(device), K=3) #recon_data will have an extra dimension, according to K
		utilslib.save_reconstructions(data, recon_data, os.path.join(GN.model_dir, "reconstructed.pdf"))
		print("Complete :)")

	elif GN.mode == "get_all_latents":

		GN.load_weights()
		GN.torch_model.eval()
		device = GN.send_model_to_device()

		GN.run_store_latent_params()


	elif GN.mode == "gradient":
		import matplotlib.pyplot as plt
		
		GN.load_weights()
		GN.torch_model.eval()
		device = GN.send_model_to_device()
		GN.run_interpolation(device, N_draw = 1, interpolation_size = 8, obj_indeces = np.array([0,21]))#path_inds, obj_distances = path_dists)


	elif GN.mode == "visualize":
		print("importing additional libraries...")
		import tqdm
		import umap
		import matplotlib.pyplot as plt
		import colorcet
		print("complete.")

		GN.load_weights()
		GN.torch_model.eval()
		device = GN.send_model_to_device()

		print("Running latent-space visualization...")
		#grab N number of data points and project their latent spaces using umap
		grab_N = 10000
		grab_N = max([(grab_N//GN.config.BATCH_SIZE) * GN.config.BATCH_SIZE, GN.config.BATCH_SIZE])
		all_latents = np.zeros((grab_N, GN.config.model_opts['VAE_IM']['latent_dim']))
		all_targets = np.zeros(grab_N)
		iter_gen = iter(GN.train_gen)
		with tqdm.tqdm(total=max([1, grab_N//GN.config.BATCH_SIZE]), unit="batch") as tepoch:
			#tepoch.set_description(f"Eval Epoch {epoch}")
			for sample in range(max([1,grab_N//GN.config.BATCH_SIZE])):
				data,targets = next(iter_gen)
				iter_start = int(sample*GN.config.BATCH_SIZE)
				import pdb;pdb.set_trace()
				all_targets[iter_start:(iter_start+GN.config.BATCH_SIZE)] = targets
				latents = GN.torch_model.get_latents(data)
				#latents = latents.cpu().detach().numpy()
				all_latents[iter_start:(iter_start+GN.config.BATCH_SIZE),:] = latents
				tepoch.update()

		utargets = np.unique(all_targets)
		colors = colorcet.glasbey[:len(utargets)]
		colors = [tuple(int(x.lstrip('#')[i:i+2], 16)/255. for i in (0, 2, 4)) for x in colors]
		if all_latents.shape[1]>2:
			#prepare umap.
			reducer = umap.UMAP()
			print("Fitting UMAP")
			embedding = reducer.fit_transform(all_latents)
			print("Complete.")
			for i,cc in enumerate(utargets):
				good_inds = all_targets==cc
				plt.plot(embedding[good_inds,0], embedding[good_inds,1], 'o', color=colors[i], label="{}".format(int(cc)))
			print("Complete.")
			print("Saving figure...")
			plt.legend()
			plt.savefig(os.path.join(GN.model_dir, "UMAP.png"),format='png') #pick png because pdf will save every dot.
		else:
			for i,cc in enumerate(utargets):
				good_inds = all_targets==cc
				plt.plot(all_latents[good_inds,0], all_latents[good_inds,1], 'o', color=colors[i], label="{}".format(int(cc)))
			print("Complete.")
			print("Saving figure...")
			plt.legend()
			plt.savefig(os.path.join(GN.model_dir, "latent_rep.png"),format='png') #pick png because pdf will save every dot.
		plt.close()

		      

	""" 
	Need to add code to visualize the latent space (Done)
	
	Need code to take 4 examples, get their latent spaces, do linear interpolation between them (all edges of a square grid)
	Then interpolate between all those. 
	Then run all those through reconstruct_from_latents
	"""

	# elif TF_net.mode == "infer":
	# 	print("NOT IMPLEMENTED YET")
	# 	pass