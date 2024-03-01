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
	if config.train_opts['classifier_loss']=="CCE":
		classification_loss = torch.nn.CrossEntropyLoss(reduction = 'sum')
	elif config.train_opts['classifier_loss']=="BCE":
		classification_loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum') #default reduction is mean.. that's fine.
	else:
		assert "", "Unrecognized loss function for training the classifier: {}".format(config.train_opts['classifier_loss'])
	#l1_crit = torch.nn.L1Loss(reduction='sum')	

	if config.train_opts['recon_loss']=="MSE":
		reconstruction_error = torch.nn.MSELoss(reduction = 'sum')


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
		epoch_class_loss = 0
		epoch_elbo_loss = 0
		epoch_kld_loss = 0
		epoch_recon_loss = 0
		epoch_class_entropy = 0
		#epoch_mask_loss = 0
		iter_n = 0

		"""
		Set random seeds appropriately!
		"""
		torch.seed()
		random.seed()
		np.random.seed()

		"""
		Train classifier in adversarial way to remove batch effects in the latent distribution and thus the reconstruction!
		"""
		if (epoch % switch_n_epochs == 0) and (epoch > 0):
			#freeze all other parameters except for the classification weights.
			for param in model.parameters():
				param.requires_grad = False
			#train the classification head weights
			for param in model.classifier.parameters():
				param.requires_grad = True

		else: #to be clear, all epochs except every nth epoch will train the classifier.
			#set all other parameters to train except for the classification weights.
			for param in model.parameters():
				param.requires_grad = True
			#freeze the classification head weights
			for param in model.classifier.parameters():
				param.requires_grad = False

		""" Train Loop """
		
		with tqdm.tqdm(total=config.STEPS_PER_EPOCH, unit="batch") as tepoch:
			#loop over random training batches

			tepoch.set_description(f"Train Epoch {epoch}")
			train_correct = 0
			train_entropy = 0
			###THE TRAINING GENERATOR SHOULD BE RANDOMLY INITIALIZED. Every call to enumerate(loader) starts from the beginning
			for train_ind,(local_batch, labels) in enumerate(training_generator): #will not loop through all, there is a break statement. 
				# import pdb;pdb.set_trace()
				# print(model.classifier.dec.classifier_dense_2.weight) # Verified, it does indeed freeze the model weights.

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
				qz_x, px_z, _, probs = model(local_batch, labels) #zs is the latent spac sampled from qz_x, recon is the reconstructed output,
				""" Probs here are actually logits, not probabilities. To get actual probabilities, you need to do a softmax! """
				true_probs = torch.nn.functional.softmax(probs, dim = 1)#.cpu().detach().numpy() #detach from GPU so we can use them in our computations.
				#qz_x, px_z, _, probs = model(local_batch) #zs is the latent spac sampled from qz_x, recon is the reconstructed output 
				
				###########################################################################################
				#####################################LOSS COMPUTATIONS#####################################
				###########################################################################################

				#################################
				##### ADVERSARIAL TRAINING ######
				#################################
				if (epoch % switch_n_epochs == 0) and (epoch > 0): #training only the classifier

					#(1) compute loss on classification head

					#labels = labels.to(device)
					kk = int(probs.shape[0] / labels.shape[0])#repeat labels for however many times the latent space was sampled!
					class_loss = classification_loss(probs, labels.repeat(kk,1))
					model_loss = config.train_opts['class_weight'] * class_loss

				else: #training the encoder/decoder to fool the discriminator. 

					#(1) compute loss on classification head

					#We want the classifier to predict everything as coming from the same sample distribution. Therefore, the ground-truth labels in this case should be all zeros, except for the target sample index. 
					#the classifier head should be frozen so the weights do not update, and in place only the latent space embedding changes to remove the sample specific stuff.
					labels = torch.ones(labels.shape) #OHE labels, should be batch size x N classes
					# labels[:,target_label] = 1.
					labels = labels / labels.sum()
					labels = labels.to(device) 
					kk = int(probs.shape[0] / labels.shape[0])#repeat labels for however many times the latent space was sampled!
					class_loss = classification_loss(probs, labels.repeat(kk,1))
					### NEW CE 11/17/23 L1 LASSO LOSS ON CLASSIFIER PARAMETERS!
					# # Prevent overfitting in the classifier model by adding L1 Regularization.
					# l1_reg_loss = 0
					# if config.train_opts['classifier_l1_weight'] > 0:
					# 	for param in model.classifier.parameters():
					# 		target = torch.autograd.Variable(torch.zeros(param.shape)).to(device)
					# 		l1_reg_loss += l1_crit(param, target)
					# # The question is, which seems to be argued, does this loss only apply itself to the classifier parameters, or does it apply itself to the whole model. 

					#(2) Compute KL loss (ELBO)

					#first term is what the probability is we would observe x given the distribution found during modeling.
					### Evidence
					if config.train_opts['recon_loss']=="MSE":
						recon = px_z.rsample()
						#import pdb;pdb.set_trace()
						recon_loss = - reconstruction_error(recon, local_batch) #needs to be negative, since always positive (Squared error) and the ELBO takes the negative of this.
					else:
						#lpx_z = px_z.log_prob(local_batch).view(px_z.batch_shape[0], -1) * model.llik_scaling ## altered, CE 08/17/23
						#lpx_z = px_z.log_prob(local_batch).view(*px_z.batch_shape[:2], -1) * model.llik_scaling #so flatten it (keep batch dimension and channel dimension) #batch_shape is a property of distribution class
						lpx_z = px_z.log_prob(local_batch[None,...]).view(*(px_z.batch_shape[:2]), -1).mean(0) * model.llik_scaling#.sum(-1).sum(0).mean() * model.llik_scaling 
						## functionally the same value as px_z.log_prob(local_batch.repeat(px_z.batch_shape[0], *[1]*len(local_batch.shape))).view(*(px_z.batch_shape[:2]), -1).sum(-1).mean(0).mean()
						#evidence that we have chosen the right model for the data. A higher value of evidence indicates we have chosen the right model...
						### This is different in appearance than most vanilla VAEs because most tend to use the MSE as the reconstruction term (evidence). MSE assumes the input data follows a normal distribution, 
						### and that the error is calculated as deviations from it. In our case, the output could be non-normal, ex. Laplacian. 
						### In the vanilla case, the output of the decoder is the "means" of the normal distribution, and therefore is exactly the pixel intensities of the reconstructed image. 
						### Surpringly, there is no redrawing from a normal distribution using these "means". The log likelihiood, or log of the PDF of a normal evaluated at matrix x, has 3 terms, and assuming the variance is fixed, 
						### two of the terms are constants, the third is the Sum of squared error multiplied by a constant. Therefore, assuming you use a fixed width error, it only serves to modulate the 
						### size of the error differences. Setting the width to 1 is acceptable in such a case. 
						#recon_loss = lpx_z.sum(-1).mean(0).sum() #first, sum the spatial dimensions, mean over batch, then sum over each channel's error.
						# recon_loss = lpx_z.mean(-1).sum(0) #first, mean over the flattened spatial and channel dimensions, sum over batch . https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
						# if all_iter==1:
						# 	import pdb;pdb.set_trace()
						recon_loss = lpx_z.sum() #/ (config.BATCH_SIZE * config.INPUT_CHANNEL_SIZE * config.INPUT_SPATIAL_SIZE * config.INPUT_SPATIAL_SIZE)
						#recon_loss = lpx_z.sum(1).mean(0)

					#second term says given the prior, how similar is our sample drawn from the learned posterior distribution?
					#kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)) #KL divergence ##ALWAYS NON-NEGATIVE
					# kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)).mean(0).sum() #KL divergence ##ALWAYS NON-NEGATIVE #should be batch x features initially, so summing sums up the errors, then you average over batch?
					kld = utilslib.kl_divergence(model.pz(*model.pz_params), qz_x).sum() #/ (config.BATCH_SIZE * config.model_opts['VAE_IM']['latent_dim'])
					#kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)).sum(1).mean(0)
					"""
					The order of these distributions DOES matter. we're trying to match the second distribution to the first.
					So it should be the PRIOR FIRST, the estimated posterior second. 
					"""
					
					if torch.isnan(kld):
						print("ERROR: KLD LOSS IS NAN")
						import pdb;pdb.set_trace()
					#or do we want to take the mean error across each feature, since they are independent normal distributions, and then sum those up? I think the latter is correct.

					#evidence_loss = lpx_z.mean(-1) #was sum(-1)

					# elbo_loss = (lpx_z.sum(-1) - (config.train_opts['kld_beta'] * kld.sum(-1))[:,None]).mean(0).sum()  #lpx_z WAS sum, not mean. 
					#elbo_loss = (lpx_z.sum(-1).sum(-1) - (config.train_opts['kld_beta'] * kld.sum(-1))).mean()  #CE 08/25/23
					elbo_loss = ((1 - kld_beta[all_iter]) * recon_loss) - (kld_beta[all_iter] * kld) #instead of all_iter could be iter_n

					# elbo_loss = (((1-config.train_opts['kld_beta']) * lpx_z.sum(-1)) - (config.train_opts['kld_beta'] * kld.sum(-1))[:,None]).mean(0).sum()  #lpx_z WAS sum, not mean. 
					#equation 6, https://yunfanj.com/blog/2021/01/11/ELBO.html.
					#mean over the batch dimension
					#sum over the channel dimension
					## Left term is the reconstruction accuracy, right term is the complexity of the variational posterior
					## the kld term can be thought of as a measure of the additional information required to express the posterior relative to the prior, as it approaches zero, the posterior is fully obtainable from the prior.
					# The goal is to minimize the ELBO (to bring our approximate posterior as close as possible to the true one -- kld ~ 0), minimizing the kl divergence leads to maximizing the ELBO
					## THE GOAL IS TO MINIMIZE THE NEGATIVE ELBO DURING TRAINING.

					elbo_loss = -elbo_loss ### According to https://lilianweng.github.io/posts/2018-08-12-vae/, the equation above IS the negative ELBO. 

					# #(3) Compute reconstruction loss
					# recon_loss = reconstruction_loss(recon[:,1:,...], local_batch[:,1:,...])

					# #(4) Compute mask reconstruction loss
					# mask_loss = mask_recon_loss(recon[:,0,...], local_batch[:,0,...])

					# model_loss = (config.train_opts['elbo_weight'] * elbo_loss) + (config.train_opts['mask_weight'] * mask_loss) + \
					# 	(config.train_opts['recon_weight'] * recon_loss) + (config.train_opts['class_weight'] * class_loss)
					model_loss = (config.train_opts['elbo_weight'] * elbo_loss) + (config.train_opts['class_weight'] * class_loss) # + (config.train_opts['classifier_l1_weight'] * l1_reg_loss)

				##################################################################################

				#accuracy 
				if config.dataset_opts['train_on_FMNIST']:
					train_correct += (torch.argmax(true_probs,dim=1) == labels.repeat(kk,1)).sum().item()
					iter_class_entropy = entropy(true_probs.cpu().detach().numpy(),axis=1).mean()
					train_entropy += iter_class_entropy
				else:
					train_correct += (torch.argmax(true_probs,dim=1) == torch.argmax(labels.repeat(kk,1), dim=1)).sum().item()
					iter_class_entropy = entropy(true_probs.cpu().detach().numpy(),axis=1).mean()
					train_entropy += iter_class_entropy

				# Gather the loss terms to report to tensorboard
				
				iter_class_loss = class_loss.item()
				epoch_class_loss += class_loss.item()
				epoch_class_entropy += iter_class_entropy

				if (epoch % switch_n_epochs != 0) | (epoch == 0): 
					#if adversarial, classifier head is fixed, training other layers and recording their losses.
					#otherwise, all layers are being trained
					iter_recon_loss = -recon_loss.item()
					iter_kld_loss = kld.item()
					iter_elbo_loss = elbo_loss.item()

					epoch_recon_loss += -recon_loss.item()
					epoch_elbo_loss += elbo_loss.item()
					epoch_kld_loss += kld.item()


				######################################################################
				######################################################################
				######################################################################
				""" WRITE ITERATION PROGRESS TO TENSORBOARD """
				##record training losses
				writer.add_scalar('class loss iter', iter_class_loss, all_iter)
				writer.add_scalar('class entropy iter', iter_class_entropy, all_iter)
				if (epoch % switch_n_epochs != 0) | (epoch == 0):
					#if adversarial, classifier head is fixed, training other layers and recording their losses.
					#otherwise, all layers are being trained 
					writer.add_scalar('recon loss iter', iter_recon_loss, all_iter)
					writer.add_scalar('elbo loss iter', iter_elbo_loss, all_iter)
					writer.add_scalar('kld loss iter', iter_kld_loss, all_iter)

				######################################################################
				######################################################################
				######################################################################
				model_loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
				optimizer.step()

				#### A trick used in some GANs, which require the parameters to be within a certain range, is to clip the data to a certain range. 
				#https://stackoverflow.com/questions/66258464/how-can-i-limit-the-range-of-parameters-in-pytorch
				# for p in model.parameters():
				# 	p.data.clamp_(min=-1.0, max=1.0)

				#Update progress bar
				# tepoch.set_postfix(recon_loss = iter_recon_loss, class_loss = iter_class_loss, elbo_loss = iter_elbo_loss, mask_loss = iter_mask_loss)
				if (epoch % switch_n_epochs != 0) | (epoch == 0): #classifier head is fixed, training other layers and recording their losses.
					tepoch.set_postfix(class_loss = iter_class_loss, class_entropy = iter_class_entropy, elbo_loss = iter_elbo_loss, recon_loss = iter_recon_loss, kld_loss = iter_kld_loss)
				else:
					tepoch.set_postfix(class_loss = iter_class_loss, class_entropy = iter_class_entropy)

				tepoch.update()

				if train_ind == config.STEPS_PER_EPOCH - 1:
					break #break the for loop
				
		epoch_class_loss /= iter_n
		epoch_class_entropy /= iter_n
		if (epoch % switch_n_epochs != 0) | (epoch == 0): #classifier head is fixed, training other layers and recording their losses.
			epoch_recon_loss /= iter_n
			epoch_elbo_loss /= iter_n
			epoch_kld_loss /= iter_n

		epoch_train_accuracy = train_correct / (iter_n * config.BATCH_SIZE * kk)

		#print the average epoch loss.
		# print(f"Epoch {epoch}/{config.EPOCHS} *** TRAIN Losses **  Recon = {epoch_recon_loss:.3f}, Class = {epoch_class_loss:.3f}, ELBO = {epoch_elbo_loss:.3f}, Mask = {epoch_mask_loss:.3f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")
		if (epoch % switch_n_epochs != 0) | (epoch == 0): 
			#if adversarial, classifier head is fixed, training other layers and recording their losses.
			#otherwise, all layers are being trained
			print(f"Epoch {epoch}/{config.EPOCHS} *** TRAIN Losses **  CE Loss = {epoch_class_loss:.3f}, Class Entropy = {epoch_class_entropy:.3f}, ELBO = {epoch_elbo_loss:.1f}, RECON = {epoch_recon_loss:.1f}, KLD = {epoch_kld_loss:.1f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")
		else:
			print(f"Epoch {epoch}/{config.EPOCHS} *** TRAIN Losses **  CE Loss = {epoch_class_loss:.3f}, Class Entropy = {epoch_class_entropy:.3f} ::: TRAIN Acc = {epoch_train_accuracy:.6f}")

		########################################################################
		########################################################################
		"""   RUN VALIDATION LOOP   """
		########################################################################
		########################################################################
		# Validation
		val_recon_loss = 0
		val_class_loss = 0
		val_class_entropy = 0
		val_elbo_loss = 0
		val_kld_loss = 0
		#val_mask_loss = 0
		val_erecon_loss = 0
		val_eclass_loss = 0
		val_eclass_entropy = 0
		val_eelbo_loss = 0
		val_ekld_loss = 0
		#val_emask_loss = 0

		iter_n = 0
		val_correct = 0
		val_ttl_loss = 0

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
					qz_x, px_z, _, probs = model(local_batch, labels) #zs is the latent spac sampled from qz_x, recon is the reconstructed output,
					""" PROBS HERE IS LOGITS, NOT PROBABILITIES! """
					true_probs = torch.nn.functional.softmax(probs, dim = 1)	

					kk = int(probs.shape[0] / labels.shape[0]) #number of drawn samples of the latent space.
					#accuracy 
					if config.dataset_opts['train_on_FMNIST']:
						val_correct += (torch.argmax(true_probs,dim=1) == labels.repeat(kk,1)).sum().item()
					else:
						val_correct += (torch.argmax(true_probs,dim=1) == torch.argmax(labels.repeat(kk,1), dim=1)).sum().item()		

					#qz_x, px_z, _, probs = model(local_batch) #zs is the latent spac sampled from qz_x, recon is the reconstructed output, 

					#########LOSS COMPUTATIONS###########

					"""
					In the validation loop, we always would like to achieve the model where the encoder is able to fool the discriminator, and produce the best reconstructions. Therefore, 
					in the validation loop, we should do the full evaluation, but for the classifier, we want the discriminator to say yes, all its predictions say the samples all come 
					from the same target sample. 

					This also makes it so during adversarial training where the classifier is getting better at discriminating the samples, that those models are not saved. We ultimately 
					do not want them anyway.
					"""

					#(1) compute loss on classification head

					#We want the classifier to predict everything as coming from the same sample distribution. Therefore, the ground-truth labels in this case should be all zeros, except for the target sample index. 
					#the classifier head should be frozen so the weights do not update, and in place only the latent space embedding changes to remove the sample specific stuff.
					## only in the case of binary classification would we want this. And with a conditional VAE, we don't want it to be predict them all to have equal probability.
					labels = torch.ones(labels.shape) #OHE labels, should be batch size x N classes
					# labels[:,target_label] = 1.
					labels = labels / labels.sum()
					labels = labels.to(device) 
					val_class_loss = classification_loss(probs, labels.repeat(kk,1)).item()
					val_class_entropy = entropy(true_probs.cpu().detach().numpy(),axis=1).mean()
					val_eclass_entropy += val_class_entropy
					val_eclass_loss += val_class_loss 

					val_ttl_loss += config.train_opts['class_weight'] * val_class_loss

					#(2) Compute KL loss (ELBO)
					if config.train_opts['recon_loss']=="MSE":
						recon = px_z.rsample() #one draw may be bad... for the other 
						recon_loss = - reconstruction_error(recon, local_batch).item()
					else:
						#first term is what the probability is we would observe x given the distribution found during modeling.
						### Evidence
						#lpx_z = px_z.log_prob(local_batch).view(px_z.batch_shape[0], -1) * model.llik_scaling ## altered, CE 08/17/23
						#lpx_z = px_z.log_prob(local_batch).view(*px_z.batch_shape[:2], -1) * model.llik_scaling #so flatten it (keep batch dimension and channel dimension) #batch_shape is a property of distribution class
						#lpx_z = px_z.log_prob(local_batch).view(px_z.batch_shape[0], -1) * model.llik_scaling
						#lpx_z = px_z.log_prob(local_batch[None,...]).mean(0).view(px_z.batch_shape[1], -1) * model.llik_scaling 
						lpx_z = px_z.log_prob(local_batch[None,...]).view(*(px_z.batch_shape[:2]), -1).mean(0) * model.llik_scaling#.sum(-1).sum(0).mean() * model.llik_scaling
						#evidence that we have chosen the right model for the data. A higher value of evidence indicates we have chosen the right model...
						#recon_loss = lpx_z.sum(-1).mean(0).sum().item() #first, sum the spatial dimensions, mean over batch, then sum over each channel's error.
						# recon_loss = lpx_z.mean(-1).sum(0).item() #first, mean over the flattened spatial and channel dimensions, sum over batch . https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
						# recon_loss = lpx_z.sum().item()
						recon_loss = lpx_z.sum().item() #/ (config.BATCH_SIZE * config.INPUT_CHANNEL_SIZE * config.INPUT_SPATIAL_SIZE * config.INPUT_SPATIAL_SIZE)
						#recon_loss = lpx_z.sum(1).mean(0).item()

					#second term says given the prior, how similar is our sample drawn from the learned posterior distribution?
					#kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)) #KL divergence ##ALWAYS NON-NEGATIVE

					# kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)).mean(0).sum().item() #CE 08/25/23
					#kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)).sum().item() #CE 08/25/23
					kld = utilslib.kl_divergence(model.pz(*model.pz_params), qz_x).sum().item() #/ (config.BATCH_SIZE * config.model_opts['VAE_IM']['latent_dim'])
					#kld = utilslib.kl_divergence(qz_x, model.pz(*model.pz_params)).sum(1).mean(0).item()

					#val_elbo_loss = (lpx_z.sum(-1) - (config.train_opts['kld_beta'] * kld.sum(-1))[:,None]).mean(0).sum().item() #equation 6, https://yunfanj.com/blog/2021/01/11/ELBO.html.
					val_elbo_loss = recon_loss - (config.train_opts['kld_beta_mag'] * beta_norm * kld) #we keep the beta term here constant, vs training.

					#val_elbo_loss = (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum().item() #equation 6, https://yunfanj.com/blog/2021/01/11/ELBO.html.

					val_elbo_loss = -val_elbo_loss
					val_kld_loss = kld
					val_recon_loss = -recon_loss

					val_eelbo_loss += val_elbo_loss
					val_ekld_loss += val_kld_loss 
					val_erecon_loss += val_recon_loss

					## Left term is the reconstruction accuracy, right term is the complexity of the variational posterior
					## the kld term can be thought of as a measure of the additional information required to express the posterior relative to the prior, as it approaches zero, the posterior is fully obtainable from the prior.
					# The goal is to minimize the ELBO (to bring our approximate posterior as close as possible to the true one -- kld ~ 0), minimizing the kl divergence leads to maximizing the ELBO
					## THE GOAL IS TO MINIMIZE THE NEGATIVE ELBO DURING TRAINING.

					# #(3) Compute reconstruction loss
					# val_recon_loss = reconstruction_loss(recon[:,:-1,...], local_batch[:,1:,...]).item()
					# val_erecon_loss += val_recon_loss

					# #(4) Compute mask reconstruction loss
					# val_mask_loss = mask_recon_loss(recon[:,-1,...], local_batch[:,0,...]).item()
					# val_emask_loss += val_mask_loss

					# val_ttl_loss += (config.train_opts['elbo_weight'] * val_elbo_loss) + (config.train_opts['mask_weight'] * val_mask_loss) + \
					# 	(config.train_opts['recon_weight'] * val_recon_loss) + (config.train_opts['class_weight'] * val_class_loss)

					val_ttl_loss += (config.train_opts['elbo_weight'] * val_elbo_loss) + (config.train_opts['class_weight'] * val_class_loss)

					#Update progress bar
					# tepoch.set_postfix(recon_loss = recon_loss/iter_n, class_loss = class_loss/iter_n, elbo_loss = elbo_loss/iter_n, mask_loss = mask_loss/iter_n)
					tepoch.set_postfix(class_loss = val_eclass_loss/iter_n, class_entropy = val_eclass_entropy/iter_n, elbo_loss = val_eelbo_loss/iter_n, recon_loss = val_erecon_loss/iter_n, kld_loss = val_ekld_loss/iter_n)

					tepoch.update()

					if val_ind == config.VALIDATION_STEPS - 1:
						break #break the validation loop.
						

		#print the average validation loss
		val_ttl_loss = val_ttl_loss / iter_n
		val_eclass_loss = val_eclass_loss / iter_n
		val_eclass_entropy = val_eclass_entropy / iter_n
		val_ekld_loss = val_ekld_loss / iter_n 
		val_erecon_loss = val_erecon_loss / iter_n 
		val_eelbo_loss = val_eelbo_loss / iter_n

		epoch_val_accuracy = val_correct / (iter_n * config.BATCH_SIZE * kk)

		print(f"Epoch {epoch}/{config.EPOCHS} *** VAL Loss = {val_ttl_loss:.6f}, RECON = {val_erecon_loss:.1f}, KLD = {val_ekld_loss:.1f}, CE = {val_eclass_loss:.1f}, Entropy = {val_eclass_entropy:.1f} ::: VAL Acc = {epoch_val_accuracy:.6f}")

		######################################################################
		######################################################################
		######################################################################
		""" WRITE EPOCH PROGRESS TO TENSORBOARD """
		#record training losses
		## Classifier training
		writer.add_scalar('Class Loss/train', epoch_class_loss, epoch)
		writer.add_scalar('Class Entropy/train', epoch_class_entropy, epoch)
		writer.add_scalar('Class Loss/val', val_eclass_loss, epoch)
		writer.add_scalar('Class Entropy/val', val_eclass_entropy, epoch)
		#Record accuracies
		writer.add_scalar('Acc/train', epoch_train_accuracy, epoch)
		writer.add_scalar('Acc/val', epoch_val_accuracy, epoch)

		#record training losses
		writer.add_scalar('Recon Loss/train', epoch_recon_loss, epoch)
		writer.add_scalar('ELBO Loss/train', epoch_elbo_loss, epoch)
		writer.add_scalar('KLD Loss/train', epoch_kld_loss, epoch)

		#record validation losses
		writer.add_scalar('Recon Loss/val', val_erecon_loss, epoch)
		writer.add_scalar('ELBO Loss/val', val_eelbo_loss, epoch)
		writer.add_scalar('KLD Loss/val', val_ekld_loss, epoch)
		
		######################################################################
		######################################################################
		######################################################################
		if (epoch % switch_n_epochs != 0) and (epoch != 0): #classifier head is fixed, training other layers and recording their losses.
			## Do not save on epochs where we are only training the classifier head
			"""  Write checkpoint for states!  """
			## write if lower validation loss or for every 10th epoch.
			if val_ttl_loss < val_min_loss:
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