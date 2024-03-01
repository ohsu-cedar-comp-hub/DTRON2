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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils 
import torch.nn.functional as F
import mmap
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")
import time
import torch
import tqdm

def generate_offset_file(filepath, has_header=True):
	"""
	Works great, runs 874K lines in 5.73 seconds
	"""
	print("    Generating memory mapped file offsets...")
	start = time.time()
	if filepath.endswith('.gz'):
		import gzip
		my_open = gzip.open
	else:
		my_open = open
	fbuff = my_open(filepath, mode='r', encoding='utf8')
	f1_mmap = mmap.mmap(fbuff.fileno(), 0, access=mmap.ACCESS_READ)
	if has_header:
		offsets=[]#[0] #first line is always at offset 0.  #however, most of our files contain a header
	else:
		offsets=[0]
	for line_no, line in enumerate(iter(f1_mmap.readline,b'')):
		if len(line)>0: #sometimes the last line is just empty.
			offsets.append(f1_mmap.tell()) #append where the next line would start. 
	end = time.time()
	print("    total elapsed time = {} min".format((end-start)/60))
	return offsets[:-1] #we skip the last one, since it just marks the last, empty row.


class cycIF_Dataset(Dataset):
	""" Cyclic IF Dataset """

	def __init__(self, csv_file: str, num_classes: int, target_col = None, bad_cols = None, class_mapping = None):
		"""
		Inputs: 
		csv_file (string): Path to csv file
		target_col (integer): Column index where target variable is
		bad_cols (list of integers, optional): list of column indeces not representing features
		https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274
		"""
		if class_mapping is None:
			self.class_mapping = {x:x for x in range(num_classes)} # The assumption is that the labels are all provided as integers! 
		else:
			self.class_mapping = class_mapping
		self.csv_path = csv_file 
		self.num_classes = num_classes
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
			if target_col not in self.bad_cols:
				self.bad_cols.append(target_col)

	def get_all_targets(self):
		#sample_ind is the column index that contains "sample name"
		""" TARGET column could be numeric OR string in this case. We want to adjust for either """
		self.memmap.seek(self.ofs[0])
		line = self.memmap.readline()
		line =  line.decode("utf8").split(',')
		if is_float(line[self.target_col]):
			targets = np.zeros(self.__len__())
		else:
			targets = np.empty(self.__len__(), dtype="object")
		with tqdm.tqdm(total=self.__len__(), unit="index") as tepoch:
			for index in range(self.__len__()):
				self.memmap.seek(self.ofs[index])
				line = self.memmap.readline()
				line =  line.decode("utf8").split(',')#line.decode("utf8").split()[0].split(',')
				target = int(line[self.target_col]) if is_float(line[self.target_col]) else line[self.target_col] 
				#line = [np.float32(x) for i,x in enumerate(line) if i not in self.bad_cols] #doesn't scale well with large features. Can update in the future.
				#line = [np.float32(x) if i!=sample_ind else x for i,x in enumerate(line.decode("utf8").split()[0].split(','))]
				targets[index] = target
				tepoch.update()
		return targets
	
	def vis_target_dist(self, targets, target_counts):
		import seaborn as sns
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1)

		sns.barplot(x = targets, y = target_counts, ax=ax)

		ax.set_title("Class Count")
		ax.set_ylabel("Counts")
		#ax.set_xticks(range(len(xlabels)))
		#ax.set_xticklabels(xlabels, rotation=90)
		plt.subplots_adjust(bottom=0.25)

		plt.savefig("/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/class_plot.pdf",format='pdf')
		plt.close()
		print("Complete.")

		
	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()
		###################################
		if index>=self.__len__():
			raise StopIteration
		###################################
		self.memmap.seek(self.ofs[index])
		line = self.memmap.readline()
		line = line.decode("utf8").split(',')#split()[0].split(',') #if it is csv, this should work...
		if hasattr(self,'target_col'):
			#one hot encoding like.
			target = np.zeros(self.num_classes, dtype=np.float32)
			ind = int(line[self.target_col]) if is_float(line[self.target_col]) else line[self.target_col] #the target could be a string OR a integer
			ind = self.class_mapping[ind]
			try:
				target[ind] = 1. #should be an integer anyway. #now a one hot encoded array.
			except:
				print("ERROR")
				print(ind)
				print(index)
				import pdb;pdb.set_trace()
		#### The loaded list contains a string (sample name) at a particular index. Note this makes our code specific to the dataset.
		#line = [np.float32(x) if i!=sample_ind else x for i,x in enumerate(line)] #first split gets rid of the "/n", then we separate by csv.
		#for each line, we only want to return the features. so we will want to delete the last 3 entries, where the last one will be the target, 
		#which should also be returned!
		
		line = [np.float32(x) if (i not in self.bad_cols) and is_float(x) else np.float32(0.) for i,x in enumerate(line)] #all the features should be numeric; that that are strings must be thrown out, but for ease will be set to 0., which the model should learn to ignore as non-informative.
		#line = [x for i,x in enumerate(line) if i not in self.bad_cols] #doesn't scale well with large features. Can update in the future.

		# if sample_ind not in self.bad_cols: #the sample is a string, whereas everything else is an integer or float. A string cannot be converted into a float32 array.
		# 	#This case is when we are splitting the file into two files.
		# 	line = np.array(line, dtype=np.object)
		# else:
		# 	#all other times.
		line = np.array(line, dtype=np.float32)#very important to convert to array, otherwise dataloader does not make each input an example.
		if hasattr(self,'target_col'):
			return line,target
		else:
			return line

	def __len__(self):
		return len(self.ofs)

	def _get_targets(self):
		""" Use the provided class mapping to get all labels returned as integers; useful for stratification of the dataset. """
		all_targets = np.zeros(self.__len__(),dtype=int)
		for index in range(self.__len__()):
			self.memmap.seek(self.ofs[index])
			line = self.memmap.readline()
			line = line.decode("utf8").split(',')#split()[0].split(',') #if it is csv, this should work...
			ind = int(line[self.target_col]) if line[self.target_col].isnumeric() else line[self.target_col] #the target could be a string OR a integer
			ind = self.class_mapping[ind]
			all_targets[index] = ind
		return all_targets

	##########################################
	"""
	Helper functions
	"""
	##########################################
	def return_col_names(self):
		self.memmap.seek(0)
		line = self.memmap.readline()
		line = line.decode('utf8').split(",")
		line = [x for i,x in enumerate(line) if i not in self.bad_cols]
		if line[-1][-1:] == "\n":
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

def marker_dataloader(csv_filepath, batch_size: int, num_classes: int,  num_workers: int = 1, bad_cols = None, target_col = None, class_mapping = None, shuffle = False, drop_last = False, pin_memory = True):
	dset = cycIF_Dataset(csv_filepath, bad_cols=bad_cols, target_col=target_col, num_classes = num_classes, class_mapping = class_mapping)
	dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last, pin_memory = pin_memory)
	return dataloader, dset.return_col_names()

def split_marker_dataloader(csv_filepath, batch_size: int, num_classes: int, split_frac: float = 0.2, num_workers: int = 1, bad_cols = None, target_col = None, class_mapping = None, shuffle = True, stratified_split = False, drop_last = True, pin_memory = True):
	dset = cycIF_Dataset(csv_filepath, bad_cols=bad_cols, target_col=target_col, num_classes = num_classes, class_mapping = class_mapping)
	#the random split might be bad, considering that we ought to stratify based on the classes. It is however, time consuming to pull out that data.
	if not stratified_split:
		train_size = int((1 - split_frac) * dset.__len__())
		test_size = dset.__len__() - train_size
		train_dataset, test_dataset = torch.utils.data.random_split(dset, [train_size, test_size]) ### WHY NOT A STRATIFIED TRAIN SPLIT??
	else:
		train_dataset, test_dataset = stratified_train_test_split(dset, dset.__len__(), split_frac)
	
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last, pin_memory = pin_memory)
	val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last, pin_memory = pin_memory)
	return train_dataloader, val_dataloader, dset.return_col_names()

def stratified_train_test_split(dataset, data_size, testsize):
	print("    STRATIFYING DATA BY CLASSIFICATION...")
	from sklearn.model_selection import train_test_split
	targets = dataset._get_targets()
	# Stratified Sampling for train and val
	train_idx, validation_idx = train_test_split(np.arange(data_size),
												test_size=testsize,
												random_state=999,
												shuffle=True,
												stratify=targets)
	
	_, counts = np.unique(targets, return_counts=True)
	class_weights = counts / np.sum(counts)
	print("     Dataset Class Frequencies = {}".format(class_weights))

	#Subset dataset for train and val	
	train_dataset = Subset(dataset, train_idx)
	validation_dataset = Subset(dataset, validation_idx)
	return train_dataset, validation_dataset

def is_float(string):
	if string is not None:
		if string.replace(".", "").isnumeric():
			return True
		else:
			return False
	else:
		return False


if __name__ == "__main__":
	"""
	Working example below.
	"""
	import Config as configurations
	config = configurations.Config()

	""" USE TO GET CONFIG FILE CLASS MAPPING AND NUM CLASSES """
	dset = cycIF_Dataset(config.dataset_opts['train_dir'], num_classes = config.NUM_CLASSES, bad_cols=config.dataset_opts['bad_cols'], target_col=config.dataset_opts['target_col'], class_mapping = config.dataset_opts['target_map'])
	# targets = dset.get_all_targets()
	# Ts,counts = np.unique(targets, return_counts=True)

	# training_gen, validation_gen, col_names = split_marker_dataloader(config.dataset_opts['train_dir'], num_classes = config.NUM_CLASSES, 
	# 						split_frac = 0.2, batch_size = config.BATCH_SIZE, bad_cols = config.dataset_opts['bad_cols'], 
	# 						target_col = config.dataset_opts['target_col'], class_mapping = config.dataset_opts['target_map'],
	# 						shuffle=True)

	#dataloader = DataLoader(dset, batch_size=4, shuffle=False, num_workers=1) #automatically converts data to tensor
	# Split dataset into two files. 
	# dset.split_train_val(val_split=0.2, write_path="./data/test")
	#exs, targets = next(iter(dataloader))
	##one hot encode targets by
	# import torch.nn.functional as F
	# F.one_hot(targets,num_classes=5)