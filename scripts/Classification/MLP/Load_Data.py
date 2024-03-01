"""
Author: Christopher Z. Eddy
Date: 02/02/23
Purpose:
Script designed for importing data using cuDF and cuML for inputs for inference and training.
Frankly, this needs to be rewritten. Two methods worked. mmap from python, and using pyarrow to load the csv by chunks.

https://stackoverflow.com/questions/69784480/how-to-set-current-position-of-mmap-mmap-seekpos-to-beginning-of-any-nth-lin
^ has been the most useful.
"""
import os 
import numpy as np
#import pyarrow as pa
#from pyarrow.csv import open_csv
#import pandas as pd
from torch.utils.data import Dataset#, Dataloader 
#from torchvision import transforms, utils 
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

    def __init__(self, csv_file):
        """
        Inputs: 
        csv_file (string): Path to csv file
        transform (callable, optional): Optional transform to be applied on a sample.  
        https://discuss.pytorch.org/t/numpy-memmap-throttles-with-dataloader-when-available-ram-less-than-file-size/83274
        """
        self.csv_path = csv_file 
        #initialize data.
        #self.initialize_source(csv_file)
        self.ofs = generate_offset_file(csv_file)# determine offset positions #we should think about caching this somehow. 
        self.fbuff = open(csv_file, mode='r', encoding='utf8')
        self.memmap = mmap.mmap(self.fbuff.fileno(), 0, access=mmap.ACCESS_READ)
        
    def __getitem__(self,index):
        # x = np.memmap(self.csv_path,  dtype='float32', mode='r', shape=(self.num_rows,self.num_cols), 
        #               offset = int(index*self.num_cols*self.bytes_per_value))
        #x = self.train_source[index]
        # try:
        #     x = self.reader.read_next_batch().to_pandas() #can we define how large the batch is?
        # except:
        #     self.reinitialize_source()
        # return np.array(x)
        # self.source.seek(index) #seek is not line based, but rather byte based. So, index is not right.
        # self.reader = open_csv(self.source)
        # x = self.reader.read_next_batch().to_pandas()
        # return x
        #x.schema
        #use above to get names and dtypes from x = open_csv(source)
        self.memmap.seek(self.ofs[index])
        line = self.memmap.readline()
        if index>0:
            sample_ind = 160
            line = [np.float32(x) if i!=sample_ind else x for i,x in enumerate(line.decode("utf8").split()[0].split(','))] #first split gets rid of the "/n", then we separate by csv.
            #for each line, we only want to return the features. so we will want to delete the last 3 entries, where the last one will be the target, 
            #which should also be returned!
        else:
            line = line.decode('utf8').split(",")
            line[-1] = line[-1][:-1] #delete the "\n"
            #we actually shouldn't even allow this line to be returned, since it is the headers!
        return line

    def __len__(self):
        return len(self.ofs)
    
    # def initialize_source(self, train_csv_path):
    #     self.source = pa.memory_map(train_csv_path, 'r')
    #     self.reader = open_csv(self.source)
    #     #we don't have to use csv. We can use the typical "read" function too.
    #     # "Rewind" the CSV file
    #     #mmap.seek(0)
    
    # def reinitialize_source(self):
    #     self.source.seek(0)
    #     self.reader = open_csv(self.source)
    
    # def iterate_batch(self):
    #     self.data = self.reader.read_next_batch().to_pandas()


dset = cycIF_Dataset("/home/groups/CEDAR/eddyc/projects/cyc_IF/UMAP/data/Normalized Mean Intensities_processed_robust_CE_20NN_globalthresh_celltyped.csv")



# col_names = ['EPCAM - Nuclei',
#  'EPCAM - Cells',
#  'EPCAM - Rims',
#  'EPCAM - Rings',
#  'Shank3 - Nuclei',
#  'Shank3 - Cells',
#  'Shank3 - Rims',
#  'Shank3 - Rings',
#  'ADAM10 - Nuclei',
#  'ADAM10 - Cells',
#  'ADAM10 - Rims',
#  'ADAM10 - Rings',
#  'CD45 - Nuclei',
#  'CD45 - Cells',
#  'CD45 - Rims',
#  'CD45 - Rings',
#  'msNLGN1 - Nuclei',
#  'msNLGN1 - Cells',
#  'msNLGN1 - Rims',
#  'msNLGN1 - Rings',
#  'NLGN1 - Nuclei',
#  'NLGN1 - Cells',
#  'NLGN1 - Rims',
#  'NLGN1 - Rings',
#  'rbNRXN1 - Nuclei',
#  'rbNRXN1 - Cells',
#  'rbNRXN1 - Rims',
#  'rbNRXN1 - Rings',
#  'NRXN1 - Nuclei',
#  'NRXN1 - Cells',
#  'NRXN1 - Rims',
#  'NRXN1 - Rings',
#  'aSMA - Nuclei',
#  'aSMA - Cells',
#  'aSMA - Rims',
#  'aSMA - Rings',
#  'AR - Nuclei',
#  'AR - Cells',
#  'AR - Rims',
#  'AR - Rings',
#  'MAOA - Nuclei',
#  'MAOA - Cells',
#  'MAOA - Rims',
#  'MAOA - Rings',
#  'PSANCAM - Nuclei',
#  'PSANCAM - Cells',
#  'PSANCAM - Rims',
#  'PSANCAM - Rings',
#  'DCX - Nuclei',
#  'DCX - Cells',
#  'DCX - Rims',
#  'DCX - Rings',
#  'ERG - Nuclei',
#  'ERG - Cells',
#  'ERG - Rims',
#  'ERG - Rings',
#  'CMYC - Nuclei',
#  'CMYC - Cells',
#  'CMYC - Rims',
#  'CMYC - Rings',
#  'NOTCH1 - Nuclei',
#  'NOTCH1 - Cells',
#  'NOTCH1 - Rims',
#  'NOTCH1 - Rings',
#  'TUBB3 - Nuclei',
#  'TUBB3 - Cells',
#  'TUBB3 - Rims',
#  'TUBB3 - Rings',
#  'CXCR4 - Nuclei',
#  'CXCR4 - Cells',
#  'CXCR4 - Rims',
#  'CXCR4 - Rings',
#  'CK8 - Nuclei',
#  'CK8 - Cells',
#  'CK8 - Rims',
#  'CK8 - Rings',
#  'ChromA - Nuclei',
#  'ChromA - Cells',
#  'ChromA - Rims',
#  'ChromA - Rings',
#  'VIM - Nuclei',
#  'VIM - Cells',
#  'VIM - Rims',
#  'VIM - Rings',
#  'MAP2 - Nuclei',
#  'MAP2 - Cells',
#  'MAP2 - Rims',
#  'MAP2 - Rings',
#  'NCAM1 - Nuclei',
#  'NCAM1 - Cells',
#  'NCAM1 - Rims',
#  'NCAM1 - Rings',
#  'CK5 - Nuclei',
#  'CK5 - Cells',
#  'CK5 - Rims',
#  'CK5 - Rings',
#  'p53 - Nuclei',
#  'p53 - Cells',
#  'p53 - Rims',
#  'p53 - Rings',
#  'INA - Nuclei',
#  'INA - Cells',
#  'INA - Rims',
#  'INA - Rings',
#  'CD90 - Nuclei',
#  'CD90 - Cells',
#  'CD90 - Rims',
#  'CD90 - Rings',
#  'PTEN - Nuclei',
#  'PTEN - Cells',
#  'PTEN - Rims',
#  'PTEN - Rings',
#  'MAP2300X - Nuclei',
#  'MAP2300X - Cells',
#  'MAP2300X - Rims',
#  'MAP2300X - Rings',
#  'SOX2 - Nuclei',
#  'SOX2 - Cells',
#  'SOX2 - Rims',
#  'SOX2 - Rings',
#  'CD31 - Nuclei',
#  'CD31 - Cells',
#  'CD31 - Rims',
#  'CD31 - Rings',
#  'AMACR - Nuclei',
#  'AMACR - Cells',
#  'AMACR - Rims',
#  'AMACR - Rings',
#  'CK14 - Nuclei',
#  'CK14 - Cells',
#  'CK14 - Rims',
#  'CK14 - Rings',
#  'ECAD - Nuclei',
#  'ECAD - Cells',
#  'ECAD - Rims',
#  'ECAD - Rings',
#  'Ki67 - Nuclei',
#  'Ki67 - Cells',
#  'Ki67 - Rims',
#  'Ki67 - Rings',
#  'TH - Nuclei',
#  'TH - Cells',
#  'TH - Rims',
#  'TH - Rings',
#  'FYN - Nuclei',
#  'FYN - Cells',
#  'FYN - Rims',
#  'FYN - Rings',
#  'CD44 - Nuclei',
#  'CD44 - Cells',
#  'CD44 - Rims',
#  'CD44 - Rings',
#  'TGFB1 - Nuclei',
#  'TGFB1 - Cells',
#  'TGFB1 - Rims',
#  'TGFB1 - Rings',
#  'CD3 - Nuclei',
#  'CD3 - Cells',
#  'CD3 - Rims',
#  'CD3 - Rings',
#  'Sample',
#  'cluster',
#  'cell_type']
