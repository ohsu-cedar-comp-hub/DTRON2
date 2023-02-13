"""
Author: Christopher Z. Eddy
Date: 02/06/23
Purpose:
General dataset script for importing CSV data from cyclic IF intensity measurements of single cells.
Data should be a matrix where the first row contains the column headers. Each subsequent is a single cell.

This script expects 163 total columns, where the 163rd is the designated target classification,
the 162nd is the Leiden cluster assignment, and the 161st is the sample name.
"""

import os 
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 
import mmap
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")
import time
import torch

def generate_offset_file(filepath, has_header=True):
	"""
	Works great, runs 874K lines in 5.73 seconds
	"""
	print("Generating memory mapped file offsets...")
	start = time.time()
	fbuff = open(filepath, mode='r', encoding='utf8')
	f1_mmap = mmap.mmap(fbuff.fileno(), 0, access=mmap.ACCESS_READ)
	if has_header:
		offsets=[]#[0] #first line is always at offset 0.  #however, most of our files contain a header
	else:
		offsets=[0]
	for line_no, line in enumerate(iter(f1_mmap.readline,b'')):
		if len(line)>0: #sometimes the last line is just empty.
			offsets.append(f1_mmap.tell()) #append where the next line would start. 
	end = time.time()
	print("total elapsed time = {} min".format((end-start)/60))
	return offsets[:-1] #we skip the last one, since it just marks the last, empty row.


class cycIF_Dataset(Dataset):
	""" Cyclic IF Dataset """

	def __init__(self, csv_file, target_col = None, bad_cols = None):
		"""
		Inputs: 
		csv_file (string): Path to csv file
		target_col (integer): Column index where target variable is
		bad_cols (list of integers, optional): list of column indeces not representing features
		https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274
		"""
		self.csv_path = csv_file 
		self.ofs = generate_offset_file(csv_file)# determine offset positions #we should think about caching this somehow. 
		self.fbuff = open(csv_file, mode='r', encoding='utf8') #stream the csv file
		self.memmap = mmap.mmap(self.fbuff.fileno(), 0, access=mmap.ACCESS_READ) #use memmap to quickly access portions of the stream

		if bad_cols is not None:
			self.bad_cols = bad_cols #if any bad columns were designated, store them.
		else:
			self.bad_cols = []

		if target_col is not None:
			self.target_col = target_col #identify which column has the target

		if hasattr(self, 'target_col'): #add the "target" column to the columns to be deleted from features.
			self.bad_cols.append(target_col)
		
	def __getitem__(self,index):
		if torch.is_tensor(index):
			index = index.tolist()
		###################################
		if index>=self.__len__():
			raise StopIteration
		###################################
		self.memmap.seek(self.ofs[index])
		line = self.memmap.readline()
		#### The loaded list contains a string (sample name) at a particular index. Note this makes our code specific to the dataset.
		sample_ind = 160
		line = [np.float32(x) if i!=sample_ind else x for i,x in enumerate(line.decode("utf8").split()[0].split(','))] #first split gets rid of the "/n", then we separate by csv.
		#for each line, we only want to return the features. so we will want to delete the last 3 entries, where the last one will be the target, 
		#which should also be returned!
		if hasattr(self,'target_col'):
			target = int(line[self.target_col]) #should be an integer anyway.

		line = [x for i,x in enumerate(line) if i not in self.bad_cols] #doesn't scale well with large features. Can update in the future.

		if sample_ind not in self.bad_cols: #the sample is a string, whereas everything else is an integer or float. A string cannot be converted into a float32 array.
			#This case is when we are splitting the file into two files.
			line = np.array(line, dtype=np.object)
		else:
			#all other times.
			line = np.array(line, dtype=np.float32)#very important to convert to array, otherwise dataloader does not make each input an example.
		if hasattr(self,'target_col'):
			return line,target
		else:
			return line

	def __len__(self):
		return len(self.ofs)
	##########################################
	"""
	Helper functions
	"""
	##########################################
	def return_col_names(self):
		self.memmap.seek(0)
		line = self.memmap.readline()
		line = line.decode('utf8').split(",")
		line[-1] = line[-1][:-1] #delete the "\n"
		return line

	def split_train_val(self, val_split=0.2, write_out=True, write_path=None):
		"""
		Sometimes you have all the data in just a single spreadsheet. In that case, we would like to split the data into a large percent towards training, 
		and a smaller percent towards validation (val_split). 

		val_split (float): number between (0,1) indicating the percent of examples to go toward validation. (1-val_split) goes to training. 
		write_out (boolean): True if you wish to write to 2 new CSV files, train.csv and val.csv
		write_path (string, optional): path for where to save split CSV files at
		"""
		import csv
		self.bad_cols=[self.target_col] #if there were bad columns, get rid of them except for the target column.
		N_cells = self.__len__()
		#randomly select floor of val_split*Ncells
		val_inds = np.random.choice(N_cells, int(np.floor(N_cells*val_split)), replace=False)
		train_inds = [x for x in np.arange(N_cells) if x not in val_inds ]
		#doing a list comparison each time will be terribly expensive... but we only need to do it once....

		#first, is write_path given?
		if write_path is None:
			write_path = os.getcwd() # write to the current working directory. 
		#for write path, there should be two folders, 'train' and 'val'. If they don't exist, write them.
		if not os.path.isdir(os.path.join(write_path,"train")):
			os.makedirs(os.path.join(write_path,"train"))
		train_path = os.path.join(write_path,"train")
		if not os.path.isdir(os.path.join(write_path,"val")):
			os.makedirs(os.path.join(write_path,"val"))
		val_path = os.path.join(write_path,"val")

		#Write to .csv files
		print("Writing train.csv....")
		with open(os.path.join(train_path,"train.csv"), mode='w') as f:
			file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			#write the header first
			file_writer.writerow(self.return_col_names())
			#write each subsequent row. 
			for ind in train_inds:
				line,target=self.__getitem__(ind)
				file_writer.writerow(list(line)+[target])

		print("Writing val.csv....")
		with open(os.path.join(val_path,"val.csv"), mode='w') as f:
			file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			#write the header first
			file_writer.writerow(self.return_col_names())
			#write each subsequent row. 
			for ind in val_inds:
				line,target=self.__getitem__(ind)
				file_writer.writerow(list(line)+[target])



"""
Working example below.
"""
# bad_cols = [160,161] #sample, cluster
# target_col = 162 #assignment
# dset = cycIF_Dataset("./data/Normalized Mean Intensities_processed_robust_CE_20NN_globalthresh_celltyped.csv", bad_cols=bad_cols, target_col=target_col)
# dataloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=1) #automatically converts data to tensor
## Split dataset into two files. 
## dset.split_train_val(val_split=0.2, write_path="./data/test")
# exs, targets = next(iter(dataloader))
# ##one hot encode targets by
# import torch.nn.functional as F
# F.one_hot(targets,num_classes=5)