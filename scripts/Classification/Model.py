"""
Author: Christopher Z. Eddy
Date: 02/06/23
Purpose:
General MLP model to perform logistic classification 
"""

import os 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import torch.nn.functional as F

#Get the device fot training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

def linear_bn_act(in_channels, out_channels):
	#batch norm normalizes data to 0 mean and unit variance for 2D data (N, C) computed over the channel dimension
	#Rather than a fixed slope for ReLU, PReLU adds a learnable slope parameter. 
	return nn.Sequential(
		nn.Linear(in_channels, out_channels),
		nn.BatchNorm1d(out_channels),
		nn.PReLU(),
	)

class Generic_MLP(nn.Module):
	"""
	When the model is built, you should be able to use print(model) and see all the layers.
	"""
	def __init__(self, n_features):
		#n_features should be a list like [163, 1024, 512, 256, 6] where 163 is input features, 6 is output class numbers
		super(Generic_MLP,self).__init__()
		self.layers = nn.Sequential()
		for i in range(len(n_features)-2):
			self.layers.add_module('Lin_{}'.format(i+1),linear_bn_act(n_features[i],n_features[i+1]))
		self.out_logits = nn.Linear(n_features[-2], n_features[-1])
	
	def forward(self, x):
		x = self.layers(x)
		x = self.out_logits(x)
		#x = x.softmax(dim = 1) #the input to CCE is expected to be the unnormalized logits.
		#https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
		return x

	def save_model(self, filename):
		torch.save(self.state_dict(), filename)

	def load_model(self, filename, cpu=False):
		if not cpu:
			self.load_state_dict(torch.load(filename))
		else:
			self.__init__(in_features)
			self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))


def init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)


#MWE
#model = Generic_MLP([16,256,128,6])
#print(model)