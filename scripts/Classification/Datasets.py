import os 
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 
import mmap
#Ignore warnings
import warnings 
warnings.filterwarnings("ignore")
import time

def generate_offset_file(filepath):
    """
    Works great, runs 874K lines in 5.73 seconds
    """
    print("Generating memory mapped file offsets...")
    start = time.time()
    fbuff = open(filepath, mode='r', encoding='utf8')
    f1_mmap = mmap.mmap(fbuff.fileno(), 0, access=mmap.ACCESS_READ)
    offsets=[0] #first line is always at offset 0. 
    for line_no, line in enumerate(iter(f1_mmap.readline,b'')):
        offsets.append(f1_mmap.tell()) #append where the next line would start. 
    end = time.time()
    print("total elapsed time = {} min".format((end-start)/60))
    return offsets


class cycIF_Dataset(Dataset):
    """ Cyclic IF Dataset """

    def __init__(self, csv_file, bad_cols = None, target_col = None):
        """
        Inputs: 
        csv_file (string): Path to csv file
        bad_cols (list of integers, optional): list of column indeces not representing features
        target_col (integer, optional): Column index where target variable is.
        https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274
        """
        self.csv_path = csv_file 
        #initialize data.
        #self.initialize_source(csv_file)
        self.ofs = generate_offset_file(csv_file)# determine offset positions #we should think about caching this somehow. 
        self.fbuff = open(csv_file, mode='r', encoding='utf8')
        self.memmap = mmap.mmap(self.fbuff.fileno(), 0, access=mmap.ACCESS_READ)
        if bad_cols is not None:
            self.bad_cols = bad_cols 
        if target_col is not None:
            self.target_col = target_col 
            if not hasattr(self, 'bad_cols'):
                bad_cols = [target_col]
            else:
                if target_col not in self.bad_cols:
                    self.bad_cols.append(target_col)
        
    def __getitem__(self,index):
        ###################################
        #### Skip the zeroth iteration ####
        if index+2>=self.__len__():
            raise StopIteration
        else:
            index += 1
        ###################################
        self.memmap.seek(self.ofs[index])
        line = self.memmap.readline()
        #### The loaded list contains a string (sample name) at a particular index. Note this makes our code specific to the dataset.
        sample_ind = 160
        line = [np.float32(x) if i!=sample_ind else x for i,x in enumerate(line.decode("utf8").split()[0].split(','))] #first split gets rid of the "/n", then we separate by csv.
        #for each line, we only want to return the features. so we will want to delete the last 3 entries, where the last one will be the target, 
        #which should also be returned!
        if hasattr(self, 'target_col'):
            target = line[self.target_col]
        if hasattr(self, 'bad_cols'):
            line = [x for i,x in enumerate(line) if i not in self.bad_cols] #doesn't scale well with large features. Can update in the future.
        return line,target

    def __len__(self):
        return len(self.ofs)

    def return_col_names(self):
        self.memmap.seek(0)
        line = self.memmap.readline()
        line = line.decode('utf8').split(",")
        line[-1] = line[-1][:-1] #delete the "\n"
        return line


"""
Working example below.
"""
# bad_cols = [160,161] #sample, cluster
# target_col = 162 #assignment
# dset = cycIF_Dataset("/home/groups/CEDAR/eddyc/projects/cyc_IF/UMAP/data/Normalized Mean Intensities_processed_robust_CE_20NN_globalthresh_celltyped.csv", bad_cols=bad_cols, target_col=target_col)
