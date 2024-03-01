
###########################################################
######### Perform the necessary library imports ###########
###########################################################

# Plotting
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

## Reproducability ## 
# Setting the seed
np.random.seed(0)

# Import utility functions
from utils import get_mean, reparameterize, Constants

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

###########################################################################
################# Basic Encoder convolutional blocks ######################
###########################################################################

def bnconv(in_channels, out_channels, sz, padding = None):
	if padding is None:
		padding = sz//2
	return nn.Sequential(
		nn.BatchNorm2d(in_channels, eps=1e-5),
		nn.LeakyReLU(inplace=True),
		nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, sz, padding=padding)),
	)

def bnconv_AI(in_channels, out_channels, sz, padding = None):
	if padding is None:
		padding = sz//2
	return nn.Sequential(
		nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, sz, padding=padding)),
		nn.BatchNorm2d(out_channels, eps=1e-5),
		nn.LeakyReLU(inplace=True),
	)

class bnconv_AI2(nn.Module):
	def __init__(self, in_channels, out_channels, sz, padding = None, activation = nn.LeakyReLU(inplace=True)):
		super().__init__()
		if padding is None:
			padding = sz//2
		self.in_channels = in_channels
		self.conv_norm = nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, sz, padding=padding))
		self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-5)
		self.activation = activation

	def forward(self, x):
		# if torch.isnan(x).sum()>0.:
		# 	print("bnconv_AI is NaN at input F = {}".format(self.in_channels))
		#if torch.isnan(x).sum()>0.:
		#	print("bnconv_AI is NaN at F = {}".format(self.in_channels))
		x = self.conv_norm(x)
		# if torch.isnan(x).sum()>0.:
		# 	print("bnconv_AI is NaN AFTER convolution/norm at F = {}".format(self.in_channels))
		x = self.batch_norm(x)
		# if torch.isnan(x).sum()>0.:
		# 	print("bnconv_AI is NaN after batch norm at F = {}".format(self.in_channels))
		if self.activation is not None:
			x = self.activation(x)
			# if torch.isnan(x).sum()>0.:
			# 	print("bnconv_AI is NaN after activation at F = {}".format(self.in_channels))
		
		return x

def identity_block(in_channels, out_channels, sz=1):
	return nn.Sequential(
		nn.BatchNorm2d(in_channels, eps=1e-5),
		nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, sz, padding=sz//2)),
	)

class res_conv_block(nn.Module):
	def __init__(self, in_channels, out_channels, sz = 1):
		super().__init__()
		self.conv = bnconv(in_channels, out_channels, sz = sz)
		self.proj = identity_block(in_channels, out_channels, sz = 1)
	
	def forward(self, x):
		x = self.proj(x) + self.conv(x)
		return x
		
class down_block(nn.Module):
	def __init__(self, in_channels, out_channels, sz=3):
		super().__init__()
		self.identity = identity_block(in_channels, out_channels, sz = 1)
		self.resconv1 = res_conv_block(in_channels, out_channels, sz = 3)
		self.resconv2 = res_conv_block(out_channels, out_channels, sz = 3)
		
	def forward(self, x):
		xself = self.identity(x)
		x = self.resconv1(x)
		x = self.resconv2(x)
		x = torch.add(x, xself)
		return xself

	
class conv_tower(nn.Module):
	def __init__(self, in_channels: int, out_channels: list, sz: int = 3, pool_size = 3):
		super().__init__()
		self.tower = nn.Sequential()
		for n in range(len(out_channels)):
			if n==0:
				self.tower.add_module("conv_tow_conv_%d"%n, down_block(in_channels, out_channels[0], sz))
			else:
				self.tower.add_module("conv_tow_conv_%d"%n, down_block(out_channels[n-1], out_channels[n], sz))

			if pool_size is not None:
				self.tower.add_module("conv_tow_pool_%d"%n, nn.AvgPool2d(kernel_size = pool_size, stride = pool_size)) #changed from max to avg 09/11/23
	
	def forward(self, x):
		x = self.tower(x)
		return x
	
########### Flattening out the encoded data ##############
	
def strided_reduction_block(in_channels, out_channels, sz: int, stride: int):
	return nn.Sequential(
		nn.BatchNorm2d(in_channels, eps=1e-5),
		nn.LeakyReLU(inplace=True),
		nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size = sz, stride = stride, padding=0)),
	)
	
def strided_reduction_block_AI(in_channels, out_channels, sz: int, stride: int):
	return nn.Sequential(
		nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size = sz, stride = stride, padding=0)),
		nn.BatchNorm2d(out_channels, eps=1e-5),
		nn.LeakyReLU(inplace=True),
	)

class reduct_fc(nn.Module):
	def __init__(self, in_channels, spatial_size: int, out_dim: int):
		super(reduct_fc, self).__init__()
		
		self.bn = nn.BatchNorm2d(in_channels)
		self.activation = nn.LeakyReLU(inplace=True)

		self.flatten = nn.Flatten()

		self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(in_channels * spatial_size * spatial_size, out_dim))


	def forward(self, x):

		#apply batch norm and activation, then flatten it out, then apply fully connected layer
		x = self.bn(x)
		x = self.activation(x)
		x = self.flatten(x)
		x = self.fc(x)

		return x
	
class reduct_fc_AI(nn.Module):
	def __init__(self, in_channels, spatial_size: int, out_dim: int):
		super(reduct_fc_AI, self).__init__()
		
		self.flatten = nn.Flatten()

		self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(in_channels * spatial_size * spatial_size, out_dim))


	def forward(self, x):
		#apply batch norm and activation, then flatten it out, then apply fully connected layer
		x = self.flatten(x)
		x = self.fc(x)

		return x


################################

class DownLayerResidual(nn.Module):
	def __init__(self, ch_in, ch_out, sz):
		super(DownLayerResidual, self).__init__()

		self.bypass = nn.Sequential(
			nn.AvgPool2d(2, stride=2, padding=0), #do the downsample
			nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
			nn.BatchNorm2d(ch_out, eps=1e-5),
			nn.LeakyReLU(inplace=True),
		)

		self.resid = nn.Sequential(
			nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride = 2, padding=1, bias=True)), #strided reduction.
			nn.BatchNorm2d(ch_in, eps=1e-5),
			nn.LeakyReLU(inplace=True),
			nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size = sz, stride = 1, padding=sz//2, bias=True)),
			nn.BatchNorm2d(ch_out, eps=1e-5),
			nn.LeakyReLU(inplace=True),
		)


	def forward(self, x):
		#print("DOWN in: min = {}, max = {}".format(torch.min(x), torch.max(x)))
		identity = self.bypass(x)
		resid = self.resid(x)
		#print("DOWN identity: min = {}, max = {}".format(torch.min(identity), torch.max(identity)))
		#print("DOWN resid: min = {}, max = {}".format(torch.min(resid), torch.max(resid)))
		x = identity + resid

		return x
	

###########################################################################
############################# ENCODER #####################################
###########################################################################

class Enc(nn.Module):
	""" Generate latent parameters for Xenium image data. """
	def __init__(self, config):
		super(Enc, self).__init__()
		layer_dims = config.model_opts['VAE_IM']['enc_layer_dims']

		self.enc = conv_tower(config.INPUT_CHANNEL_SIZE, layer_dims, sz = config.model_opts['VAE_IM']['kernel_size'], pool_size = config.model_opts['VAE_IM']['pool_size'])

		#we want to make this a linear layer where we flatten the encoder output first?
		#flatten, 
		spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(layer_dims)))
		# #combine all the spatial dimensions.
		# if config.model_opts['VAE_IM']['use_AI_model']:
		# 	self.spatial_conv = strided_reduction_block_AI(layer_dims[-1], layer_dims[-1], sz = spatial_dim, stride = spatial_dim)
		# else:
		# 	self.spatial_conv = strided_reduction_block(layer_dims[-1], layer_dims[-1], sz = spatial_dim, stride = spatial_dim) 
		## RESHAPE
		
		# # """ Totally flatten out the spatial dimension with a convolution, """
		# #bnconv(in_channels = layer_dims[-1], out_channels = layer_dims[-1], sz = spatial_dim, padding = 0)
		# #then do a pointwise convolution. No, we'll just do a linear layer
		# self.flatten = nn.Flatten()
		# #Oh geeze, if we are at spatial dim of 32x32 with 128 features, and the latent dim is 1024, each linear layer
		# #here is 134M weights. That's alot. Can we reduce this?

		self.latent_size = config.model_opts['VAE_IM']['latent_dim']

		self.reduct_fc = reduct_fc_AI(layer_dims[-1], spatial_size = spatial_dim, out_dim = 2 * self.latent_size)

		##self.c1 = strided_reduction_block(layer_dims[-1], config.model_opts['VAE_IMseq']['latent_dim'], sz = spatial_dim, stride = spatial_dim) 
		#self.c1 = nn.utils.parametrizations.spectral_norm(nn.Linear(layer_dims[-1], config.model_opts['VAE_IM']['latent_dim']))#.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
		##self.c2 = strided_reduction_block(layer_dims[-1], config.model_opts['VAE_IMseq']['latent_dim'], sz = spatial_dim, stride = spatial_dim) 
		#self.c2 = nn.utils.parametrizations.spectral_norm(nn.Linear(layer_dims[-1], config.model_opts['VAE_IM']['latent_dim']))#nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
		# c1, c2 size: latentDim x 1 x 1
		#self.noise = torch.randn(config.BATCH_SIZE,config.model_opts['VAE_IM']['latent_dim'], device = device) will create a fixed noise.

	def forward(self, x):
		x = self.enc(x)
		#e = self.spatial_conv(e) #reduce the spatial dimension entirely.
		#e = self.flatten(e)
		#loc = self.c1(e).squeeze()
		params = self.reduct_fc(x)
		loc = params[..., :self.latent_size]
		mv_std = torch.exp((F.softplus(0.5 * params[..., self.latent_size:])).clamp_(max=20))
		#return self.flatten(self.c1(e)), F.softplus(self.flatten(self.c2(e))) + Constants.eta
		#return self.c1(e).squeeze(), F.softplus(self.c2(e)).squeeze() + Constants.eta #interesting. This softplus has a near zero gradient for negative values. Why do this? It also forces the values to all be positive or zero. The added constants makes it so the distribution never has zero width.
		
		# return self.c1(e).squeeze(),  torch.exp(F.softplus(0.5 * self.c2(e)).squeeze() +  Constants.eta) ##new, CE 08/24/23 ## CHANGED CE 08/29/23 ### Changed again 09/01/23
		# noise = torch.randn_like(loc, device = device) The noise in the forward pass is ALREADY added when doing rsample vs sample.

		# mv_std = torch.exp((F.softplus(0.5 * self.c2(e)).squeeze() +  noise).clamp_(max=20))
		#mv_std = torch.exp((F.softplus(0.5 * self.c2(e)).squeeze()).clamp_(max=20))

		#we should clip the exp function to 50. When doing exp(50) = 5.18E21, which for adding 1 will have no effect.
		return loc, mv_std  ##new, CE 08/24/23 ## CHANGED CE 08/29/23 ### Changed again 09/01/23
		#the default for the latent space is to be a Gaussian distribution. The drawn variable z is then represented by 
		# z = mu + sigma*epsilon, where epsilon is random noise drawn from a normal distribution with zero mean and unit variance, and hence epsilon carries the stochasticity necessary to learn the distribution.
		# in our case, we allow the latent space to be more generalizable, that is, it can follow a distribution that is inherently non-Gaussian, perhaps Laplacian. Therefore, 
		# the parameters mu and sigma are passed as inputs into a distribution, from which we draw a latent variable z. 

class Enc_AI(nn.Module):
	""" Generate latent parameters for Xenium image data. """
	def __init__(self, config):
		super(Enc_AI, self).__init__()
		layer_dims = config.model_opts['VAE_IM']['enc_layer_dims']
		in_channels = config.INPUT_CHANNEL_SIZE
		# pool_size = config.model_opts['VAE_IM']['pool_size']
		sz = config.model_opts['VAE_IM']['kernel_size']

		self.enc = nn.Sequential()
		for n in range(len(layer_dims)):
			if n==0:
				self.enc.add_module("conv_tow_conv_%d"%n, DownLayerResidual(in_channels, layer_dims[0], sz))
			else:
				self.enc.add_module("conv_tow_conv_%d"%n, DownLayerResidual(layer_dims[n-1], layer_dims[n], sz))
			
			self.enc.add_module("conv_tow_conv_sub_%d"%n, 
				nn.Sequential(
					nn.utils.parametrizations.spectral_norm(nn.Conv2d(layer_dims[n], layer_dims[n], kernel_size = sz, stride = 1, padding=sz//2)),
					nn.BatchNorm2d(layer_dims[n], eps=1e-5),
					nn.LeakyReLU(inplace=True),
				)
			)
	

		#we want to make this a linear layer where we flatten the encoder output first?
		spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(layer_dims)))
		# #combine all the spatial dimensions.
		# self.spatial_conv = strided_reduction_block_AI(layer_dims[-1], layer_dims[-1], sz = spatial_dim, stride = spatial_dim) 
		# # """ Totally flatten out the spatial dimension with a convolution, """
		# #bnconv(in_channels = layer_dims[-1], out_channels = layer_dims[-1], sz = spatial_dim, padding = 0)
		# #then do a pointwise convolution. No, we'll just do a linear layer
		# self.flatten = nn.Flatten()
		# #Oh geeze, if we are at spatial dim of 32x32 with 128 features, and the latent dim is 1024, each linear layer
		# #here is 134M weights. That's alot. Can we reduce this?
		# #self.c1 = strided_reduction_block(layer_dims[-1], config.model_opts['VAE_IMseq']['latent_dim'], sz = spatial_dim, stride = spatial_dim) 
		# self.c1 = nn.utils.parametrizations.spectral_norm(nn.Linear(layer_dims[-1], config.model_opts['VAE_IM']['latent_dim']))#.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
		# #self.c2 = strided_reduction_block(layer_dims[-1], config.model_opts['VAE_IMseq']['latent_dim'], sz = spatial_dim, stride = spatial_dim) 
		# self.c2 = nn.utils.parametrizations.spectral_norm(nn.Linear(layer_dims[-1], config.model_opts['VAE_IM']['latent_dim']))#nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
		# # c1, c2 size: latentDim x 1 x 1
		# #self.noise = torch.randn(config.BATCH_SIZE,config.model_opts['VAE_IM']['latent_dim'], device = device) will create a fixed noise.

		self.latent_size = config.model_opts['VAE_IM']['latent_dim']

		self.reduct_fc = reduct_fc_AI(layer_dims[-1], spatial_size = spatial_dim, out_dim = 2 * self.latent_size)

	def forward(self, x):
		x = self.enc(x)
		# e = self.spatial_conv(e) #reduce the spatial dimension entirely.
		# e = self.flatten(e)
		# loc = self.c1(e).squeeze()
		# #return self.flatten(self.c1(e)), F.softplus(self.flatten(self.c2(e))) + Constants.eta
		# #return self.c1(e).squeeze(), F.softplus(self.c2(e)).squeeze() + Constants.eta #interesting. This softplus has a near zero gradient for negative values. Why do this? It also forces the values to all be positive or zero. The added constants makes it so the distribution never has zero width.
		
		# # return self.c1(e).squeeze(),  torch.exp(F.softplus(0.5 * self.c2(e)).squeeze() +  Constants.eta) ##new, CE 08/24/23 ## CHANGED CE 08/29/23 ### Changed again 09/01/23
		# # noise = torch.randn_like(loc, device = device) The noise in the forward pass is ALREADY added when doing rsample vs sample.

		# # mv_std = torch.exp((F.softplus(0.5 * self.c2(e)).squeeze() +  noise).clamp_(max=20))
		# mv_std = torch.exp((F.softplus(0.5 * self.c2(e)).squeeze()).clamp_(max=20))
		# # if torch.isnan(mv_std).sum()>0.:
		# # 	print("posterior estimated std is a NaN")
		# # if torch.isnan(loc).sum()>0.:
		# # 	print("posterior estimated mu is a NaN")

		params = self.reduct_fc(x)
		loc = params[..., :self.latent_size]
		mv_std = torch.exp((F.softplus(0.5 * params[..., self.latent_size:])).clamp_(max=20))

		#we should clip the exp function to 50. When doing exp(50) = 5.18E21, which for adding 1 will have no effect.
		return loc, mv_std  ##new, CE 08/24/23 ## CHANGED CE 08/29/23 ### Changed again 09/01/23


###########################################################################
################# Basic Decoder convolutional blocks ######################
###########################################################################

def bnconv_up(in_channels, out_channels):
	""" Increases spatial dimensionality by a factor of 2 """
	return nn.Sequential(
		nn.BatchNorm2d(in_channels, eps=1e-5),
		nn.LeakyReLU(inplace=True),
		nn.utils.parametrizations.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding = 1, output_padding = 1)), #kernel_size=2, stride=2, padding=0),
	)

class bnconv_style(nn.Module):

	def __init__(self, style_channels, in_channels, out_channels, sz: int = 3):
		super().__init__()
		self.project_style = nn.utils.parametrizations.spectral_norm(nn.Linear(style_channels, in_channels)) #style should be a shape (B, Features).
		self.conv = bnconv(in_channels, out_channels, sz = sz, padding = sz//2)

	def forward(self, x, style):
		style = self.project_style(style) #gets style to the correct dimensionality
		style = torch.unsqueeze(torch.unsqueeze(style,-1),-1) # add spatial dimensions to style.
		x = x + style #add the style to the current input
		x = self.conv(x) #combine the style and the input through convolution.
		return x

class resup_style_block(nn.Module):
	def __init__(self, in_channels, out_channels, style_channels, sz):
		super().__init__()
		self.conv0 = bnconv(in_channels, out_channels, sz )#first, do a convolution on the upscaled input. 
		self.conv1 = bnconv_style(style_channels, out_channels, out_channels, sz) #then do a convolution on the combined style and input.
		self.proj  = identity_block(in_channels, out_channels, sz=1)

	def forward(self, x, style):

		x = self.conv0(x) + self.proj(x)
		x = self.conv1(x, style)

		return x
	
class resup_block(nn.Module):
	def __init__(self, in_channels, out_channels, style_channels, sz):
		super().__init__()
		self.conv0 = bnconv(in_channels, out_channels, sz )#first, do a convolution on the upscaled input. 
		self.proj  = identity_block(in_channels, out_channels, sz=1)

	def forward(self, x, style):

		x = self.conv0(x) + self.proj(x)

		return x
	
class up_style_block(nn.Module):
	def __init__(self, in_channels, out_channels, style_channels, sz: int = 3):
		super().__init__()
		self.upsample = bnconv_up(out_channels, out_channels) #increases dimensionality by a factor of 2
		self.conv_style_block = resup_style_block(in_channels, out_channels, style_channels, sz)

	def forward(self, input):
		#input should be a tuple (x, style)
		(x, style) = input
		x = self.conv_style_block(x, style)
		x = self.upsample(x)
		return (x, style)
	
class up_block(nn.Module):
	def __init__(self, in_channels, out_channels, style_channels, sz: int = 3):
		super().__init__()
		self.upsample = bnconv_up(out_channels, out_channels) #increases dimensionality by a factor of 2
		self.conv_block = resup_block(in_channels, out_channels, style_channels, sz)

	def forward(self, input):
		#input should be a tuple (x, style)
		(x, style) = input
		x = self.conv_block(x, style)
		x = self.upsample(x)
		return (x, style)
	

class UpLayerResidual(nn.Module):
	def __init__(
		self, ch_in, ch_out, sz,
	):
		super(UpLayerResidual, self).__init__()
		self.ch_in = ch_in

		self.bypass = nn.Sequential(
			nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)),
			nn.BatchNorm2d(ch_out),
			nn.LeakyReLU(inplace=True),
			nn.Upsample(scale_factor=2),
		)

		self.resid = nn.Sequential(
			nn.utils.parametrizations.spectral_norm(
					nn.ConvTranspose2d(
						ch_in,
						ch_in,
						kernel_size=3, #was 4
						stride=2,
						padding=1,
						output_padding=1,
						bias=True,
					)
				),
			nn.BatchNorm2d(ch_in),
			nn.LeakyReLU(inplace=True),
			nn.utils.parametrizations.spectral_norm(nn.Conv2d(ch_in, ch_out, sz, stride=1, padding=sz//2, bias=True)),
			nn.BatchNorm2d(ch_out),
			nn.LeakyReLU(inplace=True)
		)
		
		#self.activation = nn.LeakyReLU(inplace=True)

	def forward(self, x):
		#print("UP in: min = {}, max = {}".format(torch.min(x), torch.max(x)))
		# if torch.isnan(x).sum()>0.:
		# 	print("input x at F = {} is NaN".format(self.ch_in))
		# 	import pdb;pdb.set_trace()
		resid = self.resid(x)
		#print("UP resid: min = {}, max = {}".format(torch.min(resid), torch.max(resid)))
		# if torch.isnan(resid).sum()>0.:
		# 	print("residual upsample at F = {} is NaN".format(self.ch_in))
		# 	import pdb;pdb.set_trace()
		identity = self.bypass(x)
		# print("UP identity: min = {}, max = {}".format(torch.min(identity), torch.max(identity)))
		# if torch.isnan(identity).sum()>0.:
		# 	print("identity upsample at F = {} is NaN".format(self.ch_in))
		# 	import pdb;pdb.set_trace()

		x = identity + resid
		#x = self.activation(x)
		# if torch.isnan(x).sum()>0.:
		# 	print("output x at F = {} is NaN".format(self.ch_in))
		# 	import pdb;pdb.set_trace()

		return x


###########################################################################
############################# DECODER #####################################
###########################################################################

class Dec(nn.Module):
	""" Generate an image given a sample from the latent space. """
	"""
	CE: We will want to pass imgChans, fBase, and latentDim in through config.
	"""
	def __init__(self, config):
		super(Dec, self).__init__()
		self.dec = nn.Sequential()
		self.layer_dims = config.model_opts['VAE_IM']['dec_layer_dims']
		style_channels = config.model_opts['VAE_IM']['latent_dim']
		sample_size = config.NUM_CLASSES #number of samples within the dataset. This will be concatenated in size to style_channels within self.reshape_fc
		output_channels = config.INPUT_CHANNEL_SIZE 

		self.spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(config.model_opts['VAE_IM']['enc_layer_dims'])-1))
		self.reshape_fc = nn.utils.parametrizations.spectral_norm(nn.Linear(style_channels + sample_size, self.layer_dims[0] * self.spatial_dim * self.spatial_dim))

		for n in range(1, len(self.layer_dims)):
			if config.model_opts['VAE_IM']['uppass_style']:
				# if n==0:
				# 	self.dec.add_module("dec_tow_res_%d"%n, up_style_block(style_channels, layer_dims[n], style_channels))
				# else:
				self.dec.add_module("dec_tow_res_%d"%n, up_style_block(self.layer_dims[n-1], self.layer_dims[n], style_channels))
			else:
				# if n==0:
				# 	self.dec.add_module("dec_tow_res_%d"%n, up_block(style_channels, layer_dims[n], style_channels))
				# else:
				self.dec.add_module("dec_tow_res_%d"%n, up_block(self.layer_dims[n-1], self.layer_dims[n], style_channels))
		
		self.final_dim_mu = bnconv(in_channels=self.layer_dims[-1], out_channels=output_channels, sz = 1)#nn.Conv2d(layer_dims[-1], output_channels, kernel_size = 1)
		#self.final_act = nn.Sigmoid()
		self.final_act = nn.Tanh()
		self.final_dim_logvar = bnconv(in_channels=self.layer_dims[-1], out_channels=output_channels, sz = 1)


	def forward(self, style, target):
		#x is data drawn from the encoder distribution, and should have shape (B x Features)
		#reshape X to the size before the last convolution in the encoder block.
		#squeeze out the weird first dimension (for some reason style has the shape 1 x B x latent_dim)
		#x = style.unsqueeze(-1).unsqueeze(-1) #add dimensions
		#apply a linear layer.
		
		#what is the shape of style here? Should be B x Features
		# target should also have shape B x config.NUM_CLASSES. We will concatenate the features
		# x = x.repeat(1, 1, self.spatial_dim, self.spatial_dim) #project data. 

		# x = self.reshape_fc(style)
		x = self.reshape_fc(torch.concat((style, target), axis=1))
		x = x.view(*style.shape[:-1], self.layer_dims[0], self.spatial_dim, self.spatial_dim)
		#input should be a tuple (x, style)
		input = (x, style)
		input = self.dec(input)
		# mu = self.final_dim_mu(input[0]) #zero argument ignores the returned "style".
		mu = self.final_act(self.final_dim_mu(input[0])) #zero argument ignores the returned "style".
		#x = self.final_act(x) #reduce everything to the range of 0-1 # I think this should be on, means should be contained.::: LET IT BE OPTIMIZED

		# mv_std = torch.exp((F.softplus(0.5 * self.final_dim_logvar(input[0]))).clamp_(max=20))
		mv_std = torch.exp((F.softplus(0.5 * self.final_dim_logvar(input[0]))).clamp_(max=20))

		"""
		The top answer in this paper suggests to just use the MSE, which assumes a normal, Gaussian distribution. Laplace is fine here, but they do highlight that it is better
		to not use a fixed variance to determine the optimal value of the variance parameter. This may be something we should change.
		https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte
		"""

		"""
		According to https://openreview.net/pdf?id=r1xaVLUYuE, a fixed variance can lead to mode collapse.
		"""
		return mu, mv_std #x, torch.tensor(0.01).to(x.device) #the second return is an argument for the Laplace scale.
	
class Dec_AI(nn.Module):
	""" Generate an image given a sample from the latent space. """
	"""
	CE: We will want to pass imgChans, fBase, and latentDim in through config.
	"""
	def __init__(self, config):
		super(Dec_AI, self).__init__()
		self.dec = nn.Sequential()
		layer_dims = config.model_opts['VAE_IM']['dec_layer_dims']
		style_channels = config.model_opts['VAE_IM']['latent_dim']
		sz = config.model_opts['VAE_IM']['kernel_size']
		output_channels = config.INPUT_CHANNEL_SIZE 
		for n in range(len(layer_dims)):
			if n==0:
				self.dec.add_module("dec_tow_res_%d"%n, UpLayerResidual(style_channels, layer_dims[n], sz = sz))
			else:
				self.dec.add_module("dec_tow_res_%d"%n, UpLayerResidual(layer_dims[n-1], layer_dims[n], sz = sz))
			
			self.dec.add_module("dec_tow_conv_sub_%d"%n, 
				bnconv_AI2(in_channels=layer_dims[n], out_channels=layer_dims[n], sz = 1),
			)
		
		self.final_dim_mu = bnconv_AI2(in_channels=layer_dims[-1], out_channels=output_channels, sz = 1, activation=nn.Tanh())#nn.Conv2d(layer_dims[-1], output_channels, kernel_size = 1)
		#self.final_act = nn.Sigmoid()
		#self.final_act = nn.Tanh()
		self.final_dim_logvar = bnconv_AI2(in_channels=layer_dims[-1], out_channels=output_channels, sz = 1, activation=None)

		self.spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(config.model_opts['VAE_IM']['enc_layer_dims'])))


	def forward(self, style):
		#x is data drawn from the encoder distribution, and should have shape (B x Features)
		#reshape X to the size before the last convolution in the encoder block.
		#squeeze out the weird first dimension (for some reason style has the shape 1 x B x latent_dim)
		# if torch.isnan(style).sum()>0.:
		# 	print("estimated style is a NaN")
		x = style.unsqueeze(-1).unsqueeze(-1) #add dimensions
		x = x.repeat(1, 1, self.spatial_dim, self.spatial_dim) #project data. 
		x = self.dec(x)
		# if torch.isnan(x).sum()>0.:
		# 	print("output from decoder is a NaN")
		# mu = self.final_act(self.final_dim_mu(x)) #zero argument ignores the returned "style".
		mu = self.final_dim_mu(x) #zero argument ignores the returned "style".

		mv_std = torch.exp((F.softplus(0.5 * self.final_dim_logvar(x))).clamp_(max=20))
		# if torch.isnan(mv_std).sum()>0.:
		# 	print("likelihood estimated std is a NaN")
		# if torch.isnan(mu).sum()>0.:
		# 	print("likelihood estimated mu is a NaN")

		"""
		The top answer in this paper suggests to just use the MSE, which assumes a normal, Gaussian distribution. Laplace is fine here, but they do highlight that it is better
		to not use a fixed variance to determine the optimal value of the variance parameter. This may be something we should change.
		https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte
		"""

		"""
		According to https://openreview.net/pdf?id=r1xaVLUYuE, a fixed variance can lead to mode collapse.
		"""
		return mu, mv_std #x, torch.tensor(0.01).to(x.device) #the second return is an argument for the Laplace scale.
	
	

###########################################################################
###################### CLASSIFICATION HEAD ################################
###########################################################################

# Classes
class classifier(nn.Module):
	""" Generate latent parameters for RNA-seq. """
	"""
	CE: Need to restructure this model. Our data is a single 1D vector, not a 2D vector like they get here from embedding. 
	Again, pass necessary arguments from the config file. 
	"""

	def __init__(self, config):
		super(classifier, self).__init__()

		layer_dims = config.model_opts['classifier']['dec_layer_dims']
		latent_dim = config.model_opts['classifier']['latent_dim']
	
		self.dec = nn.Sequential()
		for n in range(len(layer_dims)):
			#speciy layer dimensionality for dense layers
			if n==0:
				self.dec.add_module("classifier_dense_%d"%n, nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim, layer_dims[n])))
			else:
				self.dec.add_module("classifier_dense_%d"%n, nn.utils.parametrizations.spectral_norm(nn.Linear(layer_dims[n-1], layer_dims[n])))
			
			#specify activation function
			if n<len(layer_dims)-1:
				self.dec.add_module("classifier_Relu_%d"%n, nn.LeakyReLU(inplace=True))
				self.dec.add_module("classifier_drop_%d"%n, nn.Dropout(0.4))
			# else:
			# 	self.dec.add_module("classifier_softmax", nn.Softmax(dim=1))
			#### CROSS ENTROPY LOSS IS EXPECTED TO BE LOGITS, NOT PROBABILITIES!

	def forward(self, z):
		
		out = self.dec(z)
		
		return out

#############################################################
################# GAN DISCRIMINATORS ########################
#############################################################

class discriminator_binary(nn.Module):
	""" Generate latent parameters for image data. """
	def __init__(self, config):
		super(discriminator_binary, self).__init__()
		layer_dims = config.model_opts['VAE_IM']['discrim_layer_dims']

		self.enc = conv_tower(config.INPUT_CHANNEL_SIZE, layer_dims, sz = config.model_opts['VAE_IM']['discrim_kernel_size'], pool_size = config.model_opts['VAE_IM']['discrim_pool_size'])

		spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(layer_dims)))

		self.reduct_fc = reduct_fc_AI(layer_dims[-1], spatial_size = spatial_dim, out_dim = 2) #true or false logits


	def forward(self, x):
		x = self.enc(x)

		logits = self.reduct_fc(x)

		return logits

class discriminator_wasserstein(nn.Module):
	""" Generate latent parameters for image data. """
	def __init__(self, config):
		super(discriminator_binary, self).__init__()
		layer_dims = config.model_opts['VAE_IM']['discrim_layer_dims'] 

		self.enc = conv_tower(config.INPUT_CHANNEL_SIZE, layer_dims, sz = config.model_opts['VAE_IM']['discrim_kernel_size'], pool_size = config.model_opts['VAE_IM']['discrim_pool_size'])

		spatial_dim = config.INPUT_SPATIAL_SIZE//(2**(len(layer_dims)))

		self.reduct_fc = reduct_fc_AI(layer_dims[-1], spatial_size = spatial_dim, out_dim = 1) #single value representing the Wasserstein distance. Last layer is simply a linear layer


	def forward(self, x):
		x = self.enc(x)

		logits = self.reduct_fc(x)

		return logits


####################################################
############## Gland MH Network ###################
####################################################

class cycIF_VAE(nn.Module):
	""" Derive a specific sub-class of a VAE for a Image model. """

	def __init__(self, config):
		super().__init__()
		self.pz = dist.Normal#dist.Laplace # prior
		self.px_z = dist.Normal#dist.Laplace # likelihood
		self.qz_x = dist.Normal#dist.Laplace # posterior
		if config.model_opts['VAE_IM']['use_AI_model']:
			self.enc = Enc_AI(config) #encoder
			self.dec = Dec_AI(config) #decoder
		else:
			self.enc = Enc(config) #encoder
			self.dec = Dec(config) #decoder
		self.classifier = classifier(config) # classifier head
		self._qz_x_params = None  # populated in `forward`
		self.modelName = 'cycIF_VAE'
		self.llik_scaling = 1.

		grad = {'requires_grad': config.model_opts['VAE_IM']['learn_prior']}

		self._pz_params = nn.ParameterList([
			nn.Parameter(torch.zeros(1, config.model_opts['VAE_IM']['latent_dim']), requires_grad=False),  # mu
			nn.Parameter(torch.zeros(1, config.model_opts['VAE_IM']['latent_dim']), **grad)  # logvar
		]) #This means a standard normal, with mean zero and unit variance. But passing this into the softplus layer, as is shown below, returns a value of 0.6931

	@property
	def pz_params(self):
		#return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta
		return self._pz_params[0], torch.exp(F.softplus(0.5 * self._pz_params[1]) + Constants.eta) ##new, CE 08/24/23 #changed again CE 08/29/23
	
	@property
	def qz_x_params(self):
		if self._qz_x_params is None:
			raise NameError("qz_x params not initalised yet!")
		return self._qz_x_params

	def forward(self, x, target, K=5): 
		#from "A deep generative model of 3D single-cell organization" by Allen Institute, "we use a pixel-wise MSE to approximate the reconstruction likelihood and average over ten samples from z_r or z_t"
		self._qz_x_params = self.enc(x)
		qz_x = self.qz_x(*self.qz_x_params) #set the posterior distribution??
		#rsample is random sampling using reparameterization trick!
		zs = qz_x.rsample(torch.Size([K])) #draw from the posterior distribution # rsample keeps the computation graph alive, important for gradient descent.
		#squeeze the K dimension.
		#zs = zs.squeeze(0) #this is correct. 
		#zs = zs.mean(axis=0) #we don't want to apply the mean here. The mean should be applied over the sum error.
		#we should reshape it so that the drawn out dimension is mixed into the batch, we'll take it out after decoding.
		
		#reshape
		zs = zs.view((K * x.shape[0], -1))

		#compute probabilities
		probs = self.classifier(zs) # I think we should do the classificaiton on the qz_x_params at this stage, rather than a randomly drawn sample.

		#we need to repeat target K times. 
		target = target.repeat(K,1)

		params = self.dec(zs, target) # for the first argument of params, we want to only use the :-1 channels, since the last channel contains the mask.
		""" 
		The evidence part of the ELBO loss measures how accurately the network constructs the semantic output by using the likelihood distribution, forming a distance between gt X and reconstructed X.
		As such, we may not need to pull out the mask. However, the mask in the input is in the first channel. 
		"""
		#params = list(params) #params was previously a tuple, and a tuple is immutable, which doesn't work for what I need to do next.
		#M = params[0][:,-1,...] # pull out the generated mask.
		#params[0] = params[0][:,:-1,...] #now get rid of the mask channel in parameters

		#reshape params 
		params = [zz.view(K, x.shape[0], *zz.shape[1:]) for zz in params]

		if self.px_z.__name__ == "NegativeBinomial":
			#params = reparameterize(*self.dec(zs))
			px_z = self.px_z(*reparameterize(*params))#*self.dec(zs)))
		else:
			px_z = self.px_z(*params)

		return qz_x, px_z, zs, probs #return posterior distribution, 

	def generate(self, N = 1, K=1):
		self.eval()
		with torch.no_grad():
			pz = self.pz(*self.pz_params)
			#import pdb;pdb.set_trace()
			latents = pz.sample(torch.Size([N]))
			#squeeze the K dimension.
			#latents = latents.squeeze(1) #this is correct. 
			#latents = latents.mean(axis=0)
			if self.px_z.__name__ == "NegativeBinomial":
				#params = reparameterize(*self.dec(zs))
				px_z = self.px_z(*reparameterize(*self.dec(latents))) #necessary for the negative binomial
			else:
				px_z = self.px_z(*self.dec(latents))
			#import pdb;pdb.set_trace()
			#data = px_z.sample(torch.Size([K]))#px_z.sample(torch.Size([1])).squeeze(0)
			#data = data.mean(axis=0)
			recon = get_mean(px_z)
		return recon#data
	
	def get_latents(self, data, K=5):
		self.eval()
		with torch.no_grad():
			params = self.enc(data)
			#reshape params 
			params = [zz.view(K, data.shape[0], *zz.shape[1:]) for zz in params]
			qz_x = self.qz_x(*params)
			#latents = qz_x.rsample()  # no dim expansion
			samples = qz_x.sample(torch.Size([K]))
			samples = samples.mean(axis=0)
		return samples
	
	def get_posterior_params(self, data):
		self.eval()
		with torch.no_grad():
			params = self.enc(data)
		return params

	def reconstruct(self, data, target, K=1):
		self.eval()
		with torch.no_grad():
			params = self.enc(data)
			# #reshape params 
			# params = [zz.view(K, data.shape[0], *zz.shape[1:]) for zz in params]
			qz_x = self.qz_x(*params)
			latents = qz_x.sample(torch.Size([K])) # no dim expansion
			#reshape
			latents = latents.view((K * data.shape[0], -1))
			#we need to repeat target K times. 
			target = target.repeat(K,1)
			#latents = latents.mean(0) #this is correct.
			params = self.dec(latents, target)
			#reshape params 
			params = [zz.view(K, data.shape[0], *zz.shape[1:]) for zz in params]

			if self.px_z.__name__ == "NegativeBinomial":
				px_z = self.px_z(*reparameterize(*params)) #necessary for the negative binomial
			else:
				px_z = self.px_z(*params)

			recon = get_mean(px_z)
		return recon, latents
	
	def reconstruct_from_latents(self, latents):
		self.eval()
		with torch.no_grad():
			if self.px_z.__name__ == "NegativeBinomial":
				#params = reparameterize(*self.dec(zs))
				px_z = self.px_z(*reparameterize(*self.dec(latents))) #necessary for the negative binomial
			else:
				px_z = self.px_z(*self.dec(latents))

			recon = get_mean(px_z)
		return recon

	# def analyse(self, data, K):
	# 	self.eval()
	# 	with torch.no_grad():
	# 		qz_x, _, zs = self.forward(data, K=K)
	# 		pz = self.pz(*self.pz_params)
	# 		zss = [pz.sample(torch.Size([K, data.size(0)])).view(-1, pz.batch_shape[-1]),
	# 			   zs.view(-1, zs.size(-1))]
	# 		zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
	# 		kls_df = tensors_to_df(
	# 			[kl_divergence(qz_x, pz).cpu().numpy()],
	# 			head='KL',
	# 			keys=[r'KL$(q(z|x)\,||\,p(z))$'],
	# 			ax_names=['Dimensions', r'KL$(q\,||\,p)$']
	# 		)
	# 	return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
	# 		torch.cat(zsl, 0).cpu().numpy(), \
	# 		kls_df
	



if __name__ == "__main__":
	import config as configlib 
	import utils as utilslib
	config = configlib.Config()
	XR = cycIF_VAE(config)

	print("Initializing model parameters to max range -{} to {}".format(0.08, 0.08))
	# Apply Xavier initialization to all layers of your model
	XR.apply(utilslib.xavier_init_range_all_layers)
	#enc = Enc(config)

	from torchsummary import summary

	I = torch.randn(2, config.INPUT_CHANNEL_SIZE, config.INPUT_SPATIAL_SIZE, config.INPUT_SPATIAL_SIZE)
	good=False
	while not good:
		ts = torch.randint(config.NUM_CLASSES, size = (I.shape[0],1))
		if len(ts.unique())>1:
			good=True
	#OHE targets.
	targets = torch.zeros(I.shape[0], config.NUM_CLASSES)
	for b in range(I.shape[0]):
		targets[b,ts[b][0]] = 1.

	#summary(XR, I)

	# print("\n")

	K=2
	qz_x_params = XR.enc(I)
	qz_x = XR.qz_x(*qz_x_params) #set the posterior distribution??
	zs = qz_x.rsample(torch.Size([K]))

	#reshape
	zs = zs.view((K * I.shape[0], -1))

	#we need to repeat target K times. 
	targets= targets.repeat(K,1)
	# zs = zs.squeeze()
	# params = XR.dec(zs)
	# summary(XR.dec, zs[0])
	params = XR.dec(zs, targets)

	### CURRENT OUTPUT: CE 08/08/23
	#I.shape
	## torch.Size([1, 10, 256, 256])

	# ==========================================================================================
	# Layer (type:depth-idx)                   Output Shape              Param #
	# ==========================================================================================
	# ├─Enc: 1-1                               [-1]                      --
	# |    └─conv_tower: 2-1                   [-1, 256, 16, 16]         --
	# |    |    └─Sequential: 3-1              [-1, 256, 16, 16]         1,352,732
	# |    └─Sequential: 2-2                   [-1, 256, 1, 1]           --
	# |    |    └─BatchNorm2d: 3-2             [-1, 256, 16, 16]         512
	# |    |    └─ReLU: 3-3                    [-1, 256, 16, 16]         --
	# |    |    └─Conv2d: 3-4                  [-1, 256, 1, 1]           16,777,472
	# |    └─Flatten: 2-3                      [-1, 256]                 --
	# |    └─Linear: 2-4                       [-1, 1024]                263,168
	# |    └─Linear: 2-5                       [-1, 1024]                263,168
	# ├─Dec: 1-2                               [-1, 11, 256, 256]        --
	# |    └─Sequential: 2-6                   [-1, 32, 256, 256]        --
	# |    |    └─up_block: 3-5                [-1, 256, 32, 32]         3,741,952
	# |    |    └─up_block: 3-6                [-1, 128, 64, 64]         673,920
	# |    |    └─up_block: 3-7                [-1, 64, 128, 128]        201,792
	# |    |    └─up_block: 3-8                [-1, 32, 256, 256]        67,104
	# |    └─Conv2d: 2-7                       [-1, 11, 256, 256]        363
	# |    └─Sigmoid: 2-8                      [-1, 11, 256, 256]        --
	# ├─classifier: 1-3                        [-1, 10]                  --
	# |    └─Sequential: 2-9                   [-1, 10]                  --
	# |    |    └─Linear: 3-9                  [-1, 256]                 262,400
	# |    |    └─ReLU: 3-10                   [-1, 256]                 --
	# |    |    └─Linear: 3-11                 [-1, 128]                 32,896
	# |    |    └─ReLU: 3-12                   [-1, 128]                 --
	# |    |    └─Linear: 3-13                 [-1, 64]                  8,256
	# |    |    └─ReLU: 3-14                   [-1, 64]                  --
	# |    |    └─Linear: 3-15                 [-1, 32]                  2,080
	# |    |    └─ReLU: 3-16                   [-1, 32]                  --
	# |    |    └─Linear: 3-17                 [-1, 10]                  330
	# |    |    └─Softmax: 3-18                [-1, 10]                  --
	# ==========================================================================================
	# Total params: 23,648,145
	# Trainable params: 23,648,145
	# Non-trainable params: 0
	# Total mult-adds (M): 99.47
	# ==========================================================================================
	# Input size (MB): 2.50
	# Forward/backward pass size (MB): 6.02
	# Params size (MB): 90.21
	# Estimated Total Size (MB): 98.73
	# ==========================================================================================
