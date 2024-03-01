import math
import torch 
import torch.nn as nn
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
# from torch.distributions.utils import logits_to_probs


def get_mean(d, K=100):
	"""
	Extract the `mean` parameter for given distribution.
	If attribute not available, estimate from samples.
	"""
	try:
		mean = d.mean
	except NotImplementedError:
		samples = d.rsample(torch.Size([K]))
		mean = samples.mean(0)
	return mean


def log_mean_exp(value, dim=0, keepdim=False):
	return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

# Classes
class Constants(object):
	eta = 1e-6
	log2 = math.log(2)
	log2pi = math.log(2 * math.pi)
	logceilc = 88  # largest cuda v s.t. exp(v) < inf
	logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


def kl_divergence(d1, d2, K=100):
	"""Computes closed-form KL if available, else computes a MC estimate."""
	#the parameters of the models should be logvariance and mean.
	""" Same as ELBO term? """
	if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
		return torch.distributions.kl_divergence(d1, d2) 
	else:
		samples = d1.rsample(torch.Size([K]))
		return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def pdist(sample_1, sample_2, eps=1e-5):
	"""Compute the matrix of all squared pairwise distances. Code
	adapted from the torch-two-sample library (added batching).
	You can find the original implementation of this function here:
	https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

	Arguments
	---------
	sample_1 : torch.Tensor or Variable
		The first sample, should be of shape ``(batch_size, n_1, d)``.
	sample_2 : torch.Tensor or Variable
		The second sample, should be of shape ``(batch_size, n_2, d)``.
	norm : float
		The l_p norm to be used.
	batched : bool
		whether data is batched

	Returns
	-------
	torch.Tensor or Variable
		Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
		``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
	if len(sample_1.shape) == 2:
		sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
	B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
	norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
	norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
	norms = (norms_1.expand(B, n_1, n_2)
			 + norms_2.transpose(1, 2).expand(B, n_1, n_2))
	distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
	return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
	indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
	# indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
	return data[indices]

def reparameterize(mean, variance):
	"""
	Convert mean and the passed variance into negative binomial inputs k and p
	k = # of successes
	p = probability of success 
	"""
	#variance = logits_to_probs(variance)
	p = 1 / (1 + (variance / mean))
	k = mean * p
	return k,p

def save_example_images(IM, save_path):
	IM = IM.cpu().detach().numpy()
	IM = np.moveaxis(IM,1,-1) #move channel to last dimension
	if IM.shape[-1]==3:
		fig, ax = plt.subplots(IM.shape[0])
		for i,I in enumerate(IM):
			ax[i].imshow(I)
			ax[i].tick_params(
				axis='both',       # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
	else:
		fig, ax = plt.subplots(IM.shape[0], IM.shape[-1])
		for i,I in enumerate(IM):
			for ch in range(I.shape[-1]):
				ax[i,ch].imshow(I[...,ch])
				ax[i,ch].tick_params(
					axis='both',       # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelbottom=False,
					labelleft=False) # labels along the bottom edge are off
	
	plt.savefig(save_path, format = 'pdf')

def save_generated_images(IM, save_path):
	IM = IM.cpu().detach().numpy()
	assert len(IM.shape)==4, "image data (arg: IM) must have 4 dimensions, but has {}".format(len(IM.shape))
	fig, ax = plt.subplots(IM.shape[0], IM.shape[1])
	for i,I in enumerate(IM):
		M = I[0]
		ax[i,0].imshow(M)
		ax[i,0].tick_params(
			axis='both',       # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			left=False,
			right=False,
			labelbottom=False,
			labelleft=False) # labels along the bottom edge are off
		II = np.moveaxis(I[1:,...], 0, -1)
		for ch in range(II.shape[-1]):
			ax[i,ch+1].imshow(II[...,ch])
			ax[i,ch+1].tick_params(
				axis='both',       # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
	
	plt.savefig(save_path, format = 'pdf')


def beta_range_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=5):
	#n_iter = total number of iterations for which to do cycling.
	#start = value where to begin beta value.
	#stop = value where to end beta value (max beta)
	#n_cycle = number of cycles of beta to repeat
	#n_cycle
	beta = np.ones(n_iter) * stop
	if n_cycle>0:
		period = n_iter/n_cycle
		step = (stop-start)/period # linear schedule

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i+c*period) < n_iter):
			beta[int(i+c*period)] = v
			v += step
			i += 1
	return beta 

def beta_range_cycle_sin(n_iter, start=0.0, stop=1.0,  n_cycle=5):
	#n_iter = total number of iterations for which to do cycling.
	#start = value where to begin beta value.
	#stop = value where to end beta value (max beta)
	#n_cycle = number of cycles of beta to repeat
	#n_cycle
	beta = np.ones(n_iter) * stop
	if n_cycle>0:
		period = n_iter/n_cycle

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i+c*period) < n_iter):
			v = np.sin((np.pi/2)*i/period) + start
			beta[int(i+c*period)] = v
			i += 1

	return beta 

def beta_range_cycle_cos(n_iter, start=0.0, stop=1.0,  n_cycle=5):
	#n_iter = total number of iterations for which to do cycling.
	#start = value where to begin beta value.
	#stop = value where to end beta value (max beta)
	#n_cycle = number of cycles of beta to repeat
	#n_cycle
	beta = np.ones(n_iter) * stop
	if n_cycle>0:
		period = n_iter/n_cycle

	for c in range(n_cycle):
		v, i = start, 0
		while v <= stop and (int(i+c*period) < n_iter):
			v = 1- np.cos((np.pi/2)*i/period) + start
			beta[int(i+c*period)] = v
			i += 1

	return beta 


def save_reconstructions(original, recon, save_path):
	original = original.cpu().detach().numpy()
	original = (original + 1) / 2
	recon = recon.cpu().detach().numpy()
	recon = (recon+1) / 2 #put into range of 0-1
	fig, ax = plt.subplots(original.shape[0], 1+recon.shape[0])
	for i in range(original.shape[0]):
		if original[0].shape[0]==3:
			ax[i,0].imshow(np.moveaxis(original[i,:,:,:],0,-1))
			for ex in range(recon.shape[0]):
				ax[i,ex+1].imshow(np.moveaxis(recon[ex,i,:,:,:],0,-1))
		elif original[0].shape[0]==4:
			ax[i,0].imshow(np.moveaxis(original[i,1:,:,:],0,-1))
			for ex in range(recon.shape[0]):
				ax[i,ex+1].imshow(np.moveaxis(recon[ex,i,1:,:,:],0,-1))
		else:
			ax[i,0].imshow(original[i,0,:,:])
			for ex in range(recon.shape[0]):
				ax[i,ex+1].imshow(recon[ex,i,0,:,:])
		for sp in range(1+recon.shape[0]):
			ax[i,sp].tick_params(
					axis='both',       # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelbottom=False,
					labelleft=False) # labels along the bottom edge are off
	#plt.tight_layout()
	plt.savefig(save_path, format='pdf')
	plt.close()
	if original[0].shape[0]==4: #if it predicted a mask, save it to the file!
		fig, ax = plt.subplots(original.shape[0], 1+recon.shape[0])
		for i in range(original.shape[0]):
			ax[i,0].imshow(original[i,0,:,:])
			for ex in range(recon.shape[0]):
				ax[i,ex+1].imshow(recon[ex,i,0,:,:])

			for sp in range(1+recon.shape[0]):
				ax[i,sp].tick_params(
						axis='both',       # changes apply to the x-axis
						which='both',      # both major and minor ticks are affected
						bottom=False,      # ticks along the bottom edge are off
						top=False,         # ticks along the top edge are off
						left=False,
						right=False,
						labelbottom=False,
						labelleft=False) # labels along the bottom edge are off
	plt.savefig(save_path[:-4] + '_mask.pdf', format='pdf')
	plt.close()

def xavier_init_range_all_layers(layer):
	if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
		if isinstance(layer, nn.Conv2d):
			# Calculate the fan-in and fan-out for the convolutional layer
			fan_in = layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
			fan_out = layer.weight.size(0) * layer.weight.size(2) * layer.weight.size(3)
		elif isinstance(layer, nn.ConvTranspose2d):
			# Calculate the fan-in and fan-out for the convolutional layer
			fan_in = layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
			fan_out = layer.weight.size(0) * layer.weight.size(2) * layer.weight.size(3)
		elif isinstance(layer, nn.Linear):
			# Calculate the fan-in and fan-out for the linear layer
			fan_in = layer.weight.size(1)
			fan_out = layer.weight.size(0)

		# Calculate the recommended range for Xavier initialization
		limit = min(0.08, np.sqrt(6 / (fan_in + fan_out)))

		# Apply Xavier initialization scaled to your desired range
		layer.weight.data.uniform_(-limit, limit)

		if layer.bias is not None:
			layer.bias.data.zero_()

def get_gradient_examples(dataset, indeces = None):
	if not isinstance(dataset, torch.utils.data.DataLoader):
		print("     Loading ")
		if indeces is None:
			#grab targets
			tt = dataset._get_targets()
			(_, indtargets) = np.unique(tt, return_index=True)
			grab_inds = indtargets[:3]
			import pdb;pdb.set_trace()
			#get those items.
			data = []
			for i in grab_inds:
				IM, _ = dataset.__getitem__(i)
				data.append(IM)
		else:
			data = []
			for i in indeces:
				IM, _ = dataset.__getitem__(i)
				data.append(IM)

		data = np.stack(data,axis=0)
		#convert to a torch array.
		data = torch.from_numpy(data)

	else:
		test = False
		counter=0
		while not test:
			counter+=1
			if counter == 10:
				print("     Might have trouble finding a draw with multiple samples...")
			data,targets = next(iter(dataset))
			#grab two different examples.
			(utargets, indtargets) = np.unique(targets, return_index=True)
			if len(utargets)>2:
				test=True
		#grab the first example of each.
		grab_inds = indtargets[:2]
		#get the latent spaces.
		data = data[grab_inds,...]
	return data