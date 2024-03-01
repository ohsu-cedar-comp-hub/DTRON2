"""
Train Script
see https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
I think this is done correctly, CE 12/06/23
One test could be on an H&E, and subsequent tests on cycIF
"""
import torch
from torch.utils.tensorboard import SummaryWriter # tensorboard writer
import tqdm # progress bar

import utils as utilslib
import numpy as np
import random
from scipy.stats import entropy


def train(training_generator, validation_generator, model, optimizer, 
			log_path, save_path, device, config, val_min_loss = 1e8):
	
	if save_path[-1]!="/":
		save_path = save_path+"/" #directory down.

	# Tensorboard writer
	writer = SummaryWriter(log_path) #need to change this.

	# Grab the adversarial switching parameter
	switch_n_epochs = config.train_opts['adversarial_switch_n']

	# Grab the target sample variable we are aiming to normalize to
	target_label = config.train_opts['target_sample_index']

	# Grab KLD Beta term cyclic annealing schedule.
	beta_norm = config.model_opts['VAE_IM']['latent_dim'] / (config.INPUT_CHANNEL_SIZE * config.INPUT_SPATIAL_SIZE * config.INPUT_SPATIAL_SIZE)
	kld_beta = utilslib.beta_range_cycle_cos(config.EPOCHS * config.BATCH_SIZE * config.STEPS_PER_EPOCH, stop = config.train_opts['kld_beta_mag'], n_cycle = config.train_opts['kld_beta_n_cycles'])
	kld_beta = kld_beta * beta_norm

	# Necessary loss functions
	#classification_loss = torch.nn.BCEWithLogitsLoss() #note, we assume the classes have equal emphasis, which must be enforced in the dataloader.
	"""
	while the easier task is to perform a binary classification (different or same), we could also do a cross entropy loss (multiclass classifier). 
	I suppose if you do a multiclass, it may learn what specifically to remove for each type of class.
	"""

	discrim_loss = torch.nn.BCEWithLogitsLoss() #want this to be a mean, not a sum!

	MSE_loss = torch.nn.MSELoss(reduction = 'sum')

	print("Beginning training...")
	# Loop over epochs
	all_iter=0
	for epoch in range(config.EPOCHS):

		########################################################################
		########################################################################
		"""   RUN TRAIN LOOP   """
		########################################################################
		########################################################################
		# Training
		iter_n = 0
		epoch_discrim_loss = 0
		epoch_mse_loss = 0
		epoch_fake_correct = 0
		epoch_real_correct = 0

		"""
		Set random seeds appropriately!
		"""
		torch.seed()
		random.seed()
		np.random.seed()

		"""
		Train classifier in adversarial way to remove batch effects in the latent distribution and thus the reconstruction!
		"""
		for param in model.parameters():
			param.requires_grad = False

		if (epoch % switch_n_epochs == 0):
			#train the discriminator!
			for param in model.discriminator.parameters():
				param.requires_grad = True

		else: #to be clear, all epochs except every nth epoch will train the classifier.
			#set all other parameters to train except for the classification weights.
			for param in model.dec.parameters():
				param.requires_grad = True

		""" Train Loop """
		
		with tqdm.tqdm(total=config.STEPS_PER_EPOCH, unit="batch") as tepoch:
			#loop over random training batches

			tepoch.set_description(f"Train Epoch {epoch}")
			###THE TRAINING GENERATOR SHOULD BE RANDOMLY INITIALIZED. Every call to enumerate(loader) starts from the beginning
			for train_ind,(local_batch, labels) in enumerate(training_generator): #will not loop through all, there is a break statement. 
				
				#### FIRST CHANNEL OF LOCAL_BATCH SHOULD BE THE MASK!!
				#### LABELS SHOULD BE N examples x K classes!
				iter_n += 1
				all_iter += 1
				# Transfer to GPU
				local_batch = local_batch.to(device)
				labels = labels.to(device)
				# Model computations
				optimizer.zero_grad()
				# Attain model outputs for the current batch
				_, px_z, _, _ = model(local_batch, labels) #zs is the latent spac sampled from qz_x, recon is the reconstructed output,
				""" Probs here are actually logits, not probabilities. To get actual probabilities, you need to do a softmax! """
				
				likelihood_batch_shape = px_z.batch_shape #this will have shape K x BATCH x CH x H x W, so for every K we want to compute the discriminator loss; the easiest method then is to wrap this into the batch dim.
				recon = utilslib.get_mean(px_z, K=5).view(likelihood_batch_shape[0] * likelihood_batch_shape[1], *(likelihood_batch_shape[2:]))

				###########################################################################################
				#####################################LOSS COMPUTATIONS#####################################
				###########################################################################################

				#################################
				##### ADVERSARIAL TRAINING ######
				#################################
				if (epoch % switch_n_epochs == 0): #training only the discriminator

					# (2) calculate the discriminator loss.

					real_logits = model.discriminator(local_batch)
					#do binary cross entropy.
					discrim_labels = torch.zeros(real_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,1] = 1.
					discrim_labels = discrim_labels.to(device) 
					dis_loss_real = discrim_loss(real_logits, discrim_labels)
					iter_real_correct = (torch.argmax(real_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()

					fake_logits = model.discriminator(recon)
					discrim_labels = torch.zeros(fake_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,0] = 1. #training to STRENGTHEN the discriminator. Loss should be large it if is mistaken!
					discrim_labels = discrim_labels.to(device) 
					dis_loss_fake = discrim_loss(fake_logits, discrim_labels)
					iter_fake_correct = (torch.argmax(fake_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()

					dis_loss = dis_loss_real + dis_loss_fake


					model_loss = config.train_opts['discriminator_weight'] * dis_loss

				else: #training the decoder to fool the discriminator. 

					# (3) calculate the discriminator loss.
					real_logits = model.discriminator(local_batch)
					#do binary cross entropy.
					discrim_labels = torch.zeros(real_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,1] = 1.
					discrim_labels = discrim_labels.to(device) 
					dis_loss_real = discrim_loss(real_logits, discrim_labels)
					iter_real_correct = (torch.argmax(real_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()

					fake_logits = model.discriminator(recon)
					#this will have shape K x BATCH x CH x H x W, so for every K we want to compute the discriminator loss.
					discrim_labels = torch.zeros(fake_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,1] = 1. #training to FOOL the discriminator. Loss should be large if it thinks it is fake!
					discrim_labels = discrim_labels.to(device) 
					dis_loss_fake = discrim_loss(fake_logits, discrim_labels)
					iter_fake_correct = (torch.argmax(fake_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()

					dis_loss = dis_loss_fake # dis_loss_real + dis_loss_fake
					### In this scenario, since the weights of the discriminator do not get updated, we only care that the decoder can produce examples that look highly realistic.
					mse_loss = MSE_loss(recon, local_batch.repeat(likelihood_batch_shape[0],*(1 for _ in range(len(local_batch.shape)))).view(likelihood_batch_shape[0] * likelihood_batch_shape[1], *(likelihood_batch_shape[2:])))
					# model_loss = (config.train_opts['elbo_weight'] * elbo_loss) + (config.train_opts['mask_weight'] * mask_loss) + \
					# 	(config.train_opts['recon_weight'] * recon_loss) + (config.train_opts['class_weight'] * class_loss)
					model_loss = (config.train_opts['discriminator_weight'] * dis_loss) + (config.train_opts['MSE_weight'] * mse_loss) #+ (config.train_opts['classifier_l1_weight'] * l1_reg_loss)
					iter_mse_loss = mse_loss.item()
					epoch_mse_loss += iter_mse_loss
				##########################################################################

				iter_discrim_loss = dis_loss_fake.item() #note, we are only recording performance on the ability of the network to recognize the generated samples.
				epoch_discrim_loss += iter_discrim_loss

				epoch_fake_correct += iter_fake_correct 
				epoch_real_correct += iter_real_correct

				######################################################################
				######################################################################
				######################################################################
				""" WRITE ITERATION PROGRESS TO TENSORBOARD """
				##record training losses
				writer.add_scalar('discrim loss iter', iter_discrim_loss, all_iter)
				writer.add_scalar('discrim fake acc', iter_fake_correct / fake_logits.shape[0], all_iter)
				writer.add_scalar('discrim real acc', iter_real_correct / real_logits.shape[0], all_iter)
				if (epoch % switch_n_epochs != 0):
					writer.add_scalar('mse loss iter', iter_mse_loss, all_iter)
				######################################################################
				######################################################################
				######################################################################
				model_loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
				optimizer.step()

				#Update progress bar
				# tepoch.set_postfix(recon_loss = iter_recon_loss, class_loss = iter_class_loss, elbo_loss = iter_elbo_loss, mask_loss = iter_mask_loss)
				if (epoch % switch_n_epochs == 0):
					tepoch.set_postfix(discrim_loss = iter_discrim_loss, fake_acc = iter_fake_correct/fake_logits.shape[0], real_acc = iter_real_correct/real_logits.shape[0])
				else:
					tepoch.set_postfix(discrim_loss = iter_discrim_loss, mse_loss = iter_mse_loss, fake_acc = iter_fake_correct/fake_logits.shape[0], real_acc = iter_real_correct/real_logits.shape[0])

				tepoch.update()

				if train_ind == config.STEPS_PER_EPOCH - 1:
					break #break the for loop
				
		epoch_discrim_loss /= iter_n
		epoch_mse_loss /= iter_n
		epoch_fake_correct /= (iter_n * fake_logits.shape[0])
		epoch_real_correct /= (iter_n * real_logits.shape[0])

		#print the average epoch loss.
		# print(f"Epoch {epoch}/{config.EPOCHS} *** TRAIN Losses **  Recon = {epoch_recon_loss:.3f}, Class = {epoch_class_loss:.3f}, ELBO = {epoch_elbo_loss:.3f}, Mask = {epoch_mask_loss:.3f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")
		print(f"Epoch {epoch}/{config.EPOCHS} *** TRAIN Losses ** DISCRIM = {epoch_discrim_loss:.1f}, MSE = {epoch_mse_loss:.1f} ::: Fake Acc = {epoch_fake_correct:.2f}, Real Acc = {epoch_real_correct:.2f}")

		########################################################################
		########################################################################
		"""   RUN VALIDATION LOOP   """
		########################################################################
		########################################################################
		val_ediscrim_loss = 0
		val_ereal_correct = 0
		val_efake_correct = 0
		val_emse_loss = 0

		iter_n = 0

		"""
		Set random seeds appropriately!
		MAKE VALIDATION REPRODUCIBLE!
		"""
		torch.manual_seed(0)
		np.random.seed(0)
		random.seed(0)
		
		with torch.set_grad_enabled(False):
			#with tqdm.tqdm(validation_generator, unit="batch") as tepoch:
			with tqdm.tqdm(total=config.VALIDATION_STEPS, unit="batch") as tepoch:

				tepoch.set_description(f"Val Epoch {epoch}")

				#for (local_batch, labels) in tepoch:
				for val_ind,(local_batch, labels) in enumerate(validation_generator):
					iter_n += 1
					# Transfer to GPU
					local_batch = local_batch.to(device)
					labels = labels.to(device)

					# Attain model outputs for the current batch
					_, px_z, _, _ = model(local_batch, labels) #zs is the latent spac sampled from qz_x, recon is the reconstructed output,
					""" PROBS HERE IS LOGITS, NOT PROBABILITIES! """
					likelihood_batch_shape = px_z.batch_shape #this will have shape K x BATCH x CH x H x W, so for every K we want to compute the discriminator loss; the easiest method then is to wrap this into the batch dim.
					recon = utilslib.get_mean(px_z, K=5).view(likelihood_batch_shape[0] * likelihood_batch_shape[1], *(likelihood_batch_shape[2:]))
					#########LOSS COMPUTATIONS###########

					"""
					In the validation loop, we always would like to achieve the model where the encoder is able to fool the discriminator, and produce the best reconstructions. Therefore, 
					in the validation loop, we should do the full evaluation, but for the classifier, we want the discriminator to say yes, all its predictions say the samples all come 
					from the same target sample. 

					This also makes it so during adversarial training where the classifier is getting better at discriminating the samples, that those models are not saved. We ultimately 
					do not want them anyway.
					"""
					
					# (3) calculate the discriminator loss.
					real_logits = model.discriminator(local_batch)
					#do binary cross entropy.
					discrim_labels = torch.zeros(real_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,1] = 1.
					discrim_labels = discrim_labels.to(device) 
					# dis_loss_real = discrim_loss(real_logits, discrim_labels)
					val_real_correct = (torch.argmax(real_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()
					val_ereal_correct += val_real_correct

					fake_logits = model.discriminator(recon)
					discrim_labels = torch.zeros(fake_logits.shape[0],2) #OHE labels, should be batch size x N classes
					discrim_labels[:,1] = 1. #Looking to FOOL the discriminator. Loss should be large if it thinks it is fake! So if we are generating realistic looking examples, loss will be small.
					discrim_labels = discrim_labels.to(device) 
					dis_loss_fake = discrim_loss(fake_logits, discrim_labels).item()
					val_fake_correct = (torch.argmax(fake_logits,dim=1) == torch.argmax(discrim_labels,dim=1)).sum().item()
					val_efake_correct += val_fake_correct

					#dis_loss = dis_loss_fake # dis_loss_real + dis_loss_fake
					### In this scenario, we want to save the model that produces high quality generated images that look real (i.e. the predictions match the provided labels that they are real), so if the loss is large, then it isn't producing good fake images

					val_ediscrim_loss += dis_loss_fake

					if (epoch % switch_n_epochs != 0):
						mse_loss = MSE_loss(recon, local_batch.repeat(likelihood_batch_shape[0],*(1 for _ in range(len(local_batch.shape)))).view(likelihood_batch_shape[0] * likelihood_batch_shape[1], *(likelihood_batch_shape[2:]))).item()
						# model_loss = (config.train_opts['elbo_weight'] * elbo_loss) + (config.train_opts['mask_weight'] * mask_loss) + \
						# 	(config.train_opts['recon_weight'] * recon_loss) + (config.train_opts['class_weight'] * class_loss)

						val_emse_loss += mse_loss

						#Update progress bar
						# tepoch.set_postfix(recon_loss = recon_loss/iter_n, class_loss = class_loss/iter_n, elbo_loss = elbo_loss/iter_n, mask_loss = mask_loss/iter_n)
						tepoch.set_postfix(discrim_loss = val_ediscrim_loss/iter_n, mse_loss = val_emse_loss / iter_n, fake_acc = val_efake_correct/(iter_n * fake_logits.shape[0]), real_acc = val_ereal_correct/(iter_n * real_logits.shape[0]))
					
					else:
						#Update progress bar
						# tepoch.set_postfix(recon_loss = recon_loss/iter_n, class_loss = class_loss/iter_n, elbo_loss = elbo_loss/iter_n, mask_loss = mask_loss/iter_n)
						tepoch.set_postfix(discrim_loss = val_ediscrim_loss/iter_n, fake_acc = val_efake_correct/(iter_n * fake_logits.shape[0]), real_acc = val_ereal_correct/(iter_n * real_logits.shape[0]))

					tepoch.update()

					if val_ind == config.VALIDATION_STEPS - 1:
						break #break the validation loop.
						

		#print the average validation loss
		val_ediscrim_loss = val_ediscrim_loss / iter_n

		val_emse_loss /= iter_n
		val_efake_correct /= (iter_n * fake_logits.shape[0])
		val_ereal_correct /= (iter_n * real_logits.shape[0])
		val_ttl_loss = (config.train_opts['discriminator_weight'] * val_ediscrim_loss) + (config.train_opts['MSE_weight'] * val_emse_loss)#(val_ediscrim_loss

		print(f"Epoch {epoch}/{config.EPOCHS} *** VAL DISCRIM = {val_ediscrim_loss:.1f}, VAL MSE = {val_emse_loss:.1f} ::: Fake Acc = {val_efake_correct:.2f}, Real Acc = {val_ereal_correct:.2f}")

		######################################################################
		######################################################################
		######################################################################
		""" WRITE EPOCH PROGRESS TO TENSORBOARD """
		
		writer.add_scalar('Discrim Loss/train', epoch_discrim_loss, epoch)
		writer.add_scalar('Discrim Loss/val', val_ediscrim_loss, epoch)
		if (epoch % switch_n_epochs != 0):
			writer.add_scalar('MSE Loss/train', epoch_mse_loss, epoch)
			writer.add_scalar('MSE Loss/val', val_emse_loss, epoch)

		writer.add_scalar('Discrim Real Acc/train', epoch_real_correct, epoch)
		writer.add_scalar('Discrim Real Acc/val', val_ereal_correct, epoch)
		
		writer.add_scalar('Discrim Fake Acc/train', epoch_fake_correct, epoch)
		writer.add_scalar('Discrim Fake Acc/val', val_efake_correct, epoch)

		######################################################################
		######################################################################
		######################################################################
		if (epoch % switch_n_epochs != 0): #classifier head is fixed, training other layers and recording their losses.
			## Do not save on epochs where we are only training the classifier head
			"""  Write checkpoint for states!  """
			## write if lower validation loss or for every 10th epoch.
			#if epoch % 5 == 0#val_ediscrim_loss < val_min_loss:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'val_ttl_loss': val_ttl_loss
				}, save_path+"Gland_net_Epoch_%d.tar"%epoch)
			val_min_loss = val_ttl_loss
		######################################################################
		######################################################################
		######################################################################
	writer.close()