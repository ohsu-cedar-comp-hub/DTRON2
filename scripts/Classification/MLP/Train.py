"""
Author: Christopher Z. Eddy
Date: 02/06/23, updated 11/08/23
Purpose:
General training script for logits-based CCE optimization and classification
"""

import torch 
import numpy as np
from torch.utils.tensorboard import SummaryWriter #tensorboard writer 

import tqdm #progress bar; equivalent to Keras/Tensorflow

def train(config, training_generator, validation_generator, model, optimizer, device, log_path, save_path, val_min_loss = 1e8):
	if save_path[-1]!="/":
		save_path = save_path+"/" #directory down
	
	#Tensorboard writer
	writer = SummaryWriter(log_path)

	############ CLASS WEIGHTS #################
	class_weights = config.dataset_opts['class_counts']
	#convert class weights from a dictionary to a 1D torch tensor of size C, where C is the number of classes.
	#the weight should also be the emphasis weight of the given class; however class_weights from config are the class balances.
	#
	class_weights = np.array([class_weights[key] for key in class_weights.keys()])

	# #we want to overemphasize the smaller class groups:
	class_weights = np.sum(class_weights) / class_weights 
	class_weights = torch.from_numpy(class_weights).to(device) #turn into a torch tensor and send to cuda device, if possible.
	############################################
	#Define Loss functions
	#bce_loss = torch.nn.BCEWithLogitsLoss() 
	cce_loss = torch.nn.CrossEntropyLoss(weight = class_weights, label_smoothing = config.LABEL_SMOOTHING) #the 0.1 is tunable. 
	l1_crit = torch.nn.L1Loss(reduction='sum')	
	"""
	Notes: 
	We should consider using class weights, since we will certainly have an unbalanced dataset.
	"""
	
	#####################################################################################
	print("Beginning training...")
	#Loop over epochs 
	all_iter=0 #will keep track of how many training iterations we have completed.
	for epoch in range(config.EPOCHS):
		######################################
		""" RUN TRAIN LOOP """
		######################################
		#Training 
		epoch_cce_loss = 0 #will accumulate the total bce loss for each batch to report an epoch average.
		epoch_l1_loss = 0
		iter_n = 0 #define how many iterations make up each epoch.

		### This below is a gross way of writing the code, but it was the only way to get the tqdm progress bar to update correctly.
		with tqdm.tqdm(total=config.STEPS_PER_EPOCH, unit="batch") as tepoch:
			#loop over training batches 
			tepoch.set_description(f"Train Epoch {epoch}")
			train_correct = 0
			#import pdb;pdb.set_trace()
			for train_ind,(local_batch,targets) in enumerate(training_generator):
				iter_n += 1
				all_iter += 1
				#transfer to GPU
				local_batch = local_batch.to(device)
				targets = targets.to(device)

				#Model computations
				optimizer.zero_grad()
				probs = model(local_batch) #these are logits, not probabilities! Use softmax to get probabilities
				loss = cce_loss(probs, targets) 
				iter_cce_loss = loss.item()
				epoch_cce_loss += iter_cce_loss

				l1_reg_loss = 0
				if config.train_opts['classifier_l1_weight'] > 0:
					for param in model.parameters():
						target = torch.autograd.Variable(torch.zeros(param.shape)).to(device)
						l1_reg_loss += l1_crit(param, target)
					loss += config.train_opts['classifier_l1_weight'] * l1_reg_loss
					epoch_l1_loss += l1_reg_loss.item()

				#accuracy 
				train_correct += (torch.argmax(torch.nn.functional.softmax(probs,dim=1),dim=1) == torch.argmax(targets, dim=1)).sum().item()

				######################################################################
				######################################################################
				""" WRITE ITERATION PROGRESS TO TENSORBOARD """
				#record training losses
				writer.add_scalar('CCE Loss iter', iter_cce_loss, all_iter)
				if config.train_opts['classifier_l1_weight'] > 0:
					writer.add_scalar('Sparsity Loss iter', l1_reg_loss.item(), all_iter)
				######################################################################
				######################################################################

				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
				optimizer.step()

				#Update progress bar
				tepoch.set_postfix(cce_loss = iter_cce_loss)
				tepoch.update()

				if train_ind == config.STEPS_PER_EPOCH - 1:
					break #break the for loop

		epoch_cce_loss /= iter_n
		if config.train_opts['classifier_l1_weight'] > 0:
			epoch_l1_loss /= iter_n
		epoch_train_accuracy = train_correct / (iter_n * config.BATCH_SIZE)

		#print the average epoch loss.
		if config.train_opts['classifier_l1_weight'] > 0:
			print(f"Epoch {epoch+1}/{config.EPOCHS} *** TRAIN CCE_Loss = {epoch_cce_loss:.6f}, TRAIN L1_Loss = {epoch_l1_loss:.6f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")
		else:
			print(f"Epoch {epoch+1}/{config.EPOCHS} *** TRAIN CCE_Loss = {epoch_cce_loss:.6f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")

		###############################
		"""   RUN VALIDATION LOOP   """
		###############################
		# Validation
		val_cce_loss = 0
		iter_n = 0
		val_correct = 0

		with torch.no_grad(): ##Same as -> #set_grad_enabled(False):
			with tqdm.tqdm(total=config.VALIDATION_STEPS, unit="batch") as tepoch:

				tepoch.set_description(f"Val Epoch {epoch}")

				for val_ind,(local_batch,targets) in enumerate(validation_generator):
					iter_n += 1
					#transfer to GPU
					local_batch = local_batch.to(device)
					targets = targets.to(device)

					#model cce loss
					probs = model(local_batch)
					val_cce_loss += cce_loss(probs, targets).item()

					#accuracy 
					val_correct += (torch.argmax(torch.nn.functional.softmax(probs,dim=1),dim=1) == torch.argmax(targets, dim=1)).sum().item()

					#we could in the future add (1) accuracy, since we can do an argmax calculation of sigma(x) where sigma is the softmax function.

					#Update progress bar 
					tepoch.set_postfix(cce_loss = val_cce_loss/iter_n)
					tepoch.update()

					if val_ind == config.VALIDATION_STEPS - 1:
						break #break the validation loop.

		epoch_val_accuracy = val_correct / (iter_n * config.BATCH_SIZE)

		#print the average validation loss
		print(f"Epoch {epoch+1}/{config.EPOCHS} *** VAL Loss = {val_cce_loss:.6f} ::: VAL Acc = {epoch_val_accuracy:.6f}")

	   
		######################################################################
		""" WRITE EPOCH PROGRESS TO TENSORBOARD """
		#record training losses
		writer.add_scalar('CCE Loss/train', epoch_cce_loss, epoch)
		if config.train_opts['classifier_l1_weight'] > 0:
			writer.add_scalar('Sparsity Loss iter', epoch_l1_loss, epoch)
		#record validation losses
		writer.add_scalar('CCE Loss/val', val_cce_loss, epoch)

		#Record accuracies
		writer.add_scalar('Acc/train', epoch_train_accuracy, epoch)
		writer.add_scalar('Acc/val', epoch_val_accuracy, epoch)
		######################################################################

		######################################################################
		"""  Write checkpoint for states!  """
		## write if lower validation loss or for every 10th epoch.
		if val_cce_loss<val_min_loss or epoch%10==0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_ttl_loss': val_cce_loss
				}, save_path+"MLP_Epoch_%d.tar"%epoch)
			val_min_loss = val_cce_loss
		######################################################################
	writer.close()