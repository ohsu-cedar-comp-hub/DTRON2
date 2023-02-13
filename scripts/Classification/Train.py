"""
Author: Christopher Z. Eddy
Date: 02/06/23
Purpose:
General training script for logits-based BCE optimization and classification
"""

import torch 
from torch.utils.tensorboard import SummaryWriter #tensorboard writer 

import tqdm #progress bar; equivalent to Keras/Tensorflow

def train(train_params, training_set, validation_set, model, optimizer, device, log_path, save_path, num_classes, max_epochs: int=50, val_min_loss = 1e8):
	if save_path[-1]!="/":
		save_path = save_path+"/" #directory down
	
	#Tensorboard writer
	writer = SummaryWriter(log_path)

	#this will be passed into train.
	#CUDA for PyTorch  ### This will be passed to the train loop.
	#use_cuda = torch.cuda.is_available()
	# device = "cuda" if torch.cuda.is_available() else "cpu"
	# print(f"Using {device} device")

	#models and optimizers are first loaded to CPU. Send to GPU.
	print("sending models to GPU device...")
	#model.cuda()
	model.to(device) #this is proper. Cuda will send it to device.
	#we don't need to send the optimizers.

	print("Creating data generators...")
	#data generators
	training_generator = torch.utils.data.DataLoader(training_set, **train_params)
	validation_generator = torch.utils.data.DataLoader(validation_set, **train_params)

	#Define Loss functions
	bce_loss = torch.nn.BCEWithLogitsLoss() 
	"""
	Notes: 
	We should consider using class weights, since we will certainly have an unbalanced dataset.
	"""
	# optimizer = torch.optim.SGD(model.paramters(), lr=learning_rate) 
	#the optimizer should be provided as an argument so that we can save its status and load it if necessary.
	
	#####################################################################################
	print("Beginning training...")
	#Loop over epochs 
	all_iter=0 #will keep track of how many training iterations we have completed.
	for epoch in range(max_epochs):
		######################################
		""" RUN TRAIN LOOP """
		######################################
		#Training 
		epoch_bce_loss = 0 #will accumulate the total bce loss for each batch to report an epoch average.
		iter_n = 0 #define how many iterations make up each epoch.

		### This below is a gross way of writing the code, but it was the only way to get the tqdm progress bar to update correctly.
		with tqdm.tqdm(training_generator, unit='batch') as tepoch:
			#loop over training batches 
			tepoch.set_description(f"Train Epoch {epoch}")

			for local_batch,targets in tepoch:
				#one hot encode targets.
				targets = torch.nn.functional.one_hot(targets, num_classes=num_classes)
				iter_n += 1
				all_iter += 1
				#transfer to GPU
				local_batch = local_batch.to(device)
				targets = targets.to(device)

				#Model computations
				optimizer.zero_grad()
				logits = model(local_batch)
				loss = bce_loss(logits, targets) 
				iter_bce_loss = loss.item()
				epoch_bce_loss += iter_bce_loss



				######################################################################
				""" WRITE ITERATION PROGRESS TO TENSORBOARD """
				#record training losses
				writer.add_scalar('BCE Loss iter', iter_bce_loss, all_iter)
				######################################################################

				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.paramters(),5.) #gradient clipping
				optimizer.step()

				#Update progress bar
				tepoch.set_postfix(bce_loss = iter_bce_loss)

		epoch_bce_loss /= iter_n 

		#print the average epoch loss.
		print(f"Epoch {epoch+1}/{max_epochs} *** TRAIN BCE_Loss = {epoch_bce_loss:.6f}")


		###############################
		"""   RUN VALIDATION LOOP   """
		###############################
		# Validation
		val_bce_loss = 0
		iter_n = 0

		with torch.no_grad(): ##Same as -> #set_grad_enabled(False):
			with tqdm.tqdm(validation_generator, unit="batch") as tepoch:

				tepoch.set_description(f"Val Epoch {epoch}")

				for local_batch,targets in tepoch:
					#one hot encode targets.
					targets = torch.nn.functional.one_hot(targets, num_classes=num_classes)
					iter_n += 1
					#transfer to GPU
					local_batch = local_batch.to(device)
					targets = targets.to(device)

					#model bce loss
					logits = model(local_batch)
					val_bce_loss += bce_loss(logits, targets).item()

					#we could in the future add (1) accuracy, since we can do an argmax calculation of sigma(x) where sigma is the softmax function.

					#Update progress bar 
					tepoch.set_postfix(bce_loss = val_bce_loss/iter_n)

		#print the average validation loss
		print(f"Epoch {epoch+1}/{max_epochs} *** VAL Loss = {val_bce_loss:.6f}")

	   
		######################################################################
		""" WRITE EPOCH PROGRESS TO TENSORBOARD """
		#record training losses
		writer.add_scalar('BCE Loss/train', epoch_bce_loss, epoch)
		#record validation losses
		writer.add_scalar('BCE Loss/val', val_bce_loss, epoch)
		######################################################################

		######################################################################
		"""  Write checkpoint for states!  """
		## write if lower validation loss or for every 10th epoch.
		if val_bce_loss<val_min_loss or epoch%10==0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_bce_loss': val_bce_loss
				}, save_path+"MLP_Epoch_%d.tar"%epoch)
			val_min_loss = val_ttl_loss
		######################################################################