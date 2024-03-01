"""
Author: Christopher Z. Eddy
Purporse: The ACED immune dataset is 13 gB large, and 7.52 Million lines long. Too much to load into memory and operate with. We know we can do the clustering, etc with 1M cells. Therefore,  I want to randomly sample this dataset to something more digestible.
"""
import numpy as np
import pandas as pd
import random

np.random.seed(0) #set for reproducibility.
random.seed(0)

N = 7519482 #skipping header size. 
##if size unknown, use
# filename = "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/ACED/ACED_immune.csv"
# N = sum(1 for line in open(filename)) - 1 #exluding header.

s = 1000000 #desired sample size. 

skip = sorted(random.sample(range(1,N+1), N-s)) # the 0-th indexed header will not be included in the skipped list.


print("Loading dataframe...")
df = pd.read_csv("/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/ACED/ACED_immune.csv", skiprows = skip)
print("Writing to file...")
df.to_csv("/home/groups/CEDAR/eddyc/projects/cyc_IF/DTRON2/data/classification/ACED_immune_random_sample.csv", index = False)
print("Complete :)")