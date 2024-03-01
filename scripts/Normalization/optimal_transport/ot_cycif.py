import numpy as np
import ot  # Python Optimal Transport library
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/Cell_Typing_041423/data/Normalized Mean Intensities_processed_robust_CE_20NN_leiden.parquet.csv",nrows = 100000)

Sample_list = ['17633-scene0-3',
 '19142-3',
 '24952-3',
 '30403-3',
 '30411-3',
 '31022-scene0-3',
 '31476-3',
 '31480-3',
 '33548-scene00-3',
 '38592-3',
 '48411-3',
 '54774-3',
 '54776-3',
 '57494-3',
 '57657-3',
 '57658-3',
 '92352-3']

labeldict = {x:y for (x,y) in zip(Sample_list, np.array(['G3', 'G3', 'G3', 'G4','TAN','TAN','G3','TAN','G3','G4','G3','G4','G4','G4', 'G4', 'G4', 'G4']))} #changed df to data

# fig, ax = plt.subplots(1)
# #show the histograms comparing, for example AR - Cells
# grp = 'ECAD - Cells'
# df.hist(grp, by='Sample', ax = ax, bins=100)
# plt.savefig("{}.pdf".format(grp),format='pdf')

#I don't think we want to do this, at least not on an inter-sample, intra-batch level. Samples can have distinct biology - some can be tumor adjacent, others G3, G4 and therefore would be expected to have very different distributions. 
# So OT cannot be used to remove the 'inter-sample' effect and must be handled differently.

#What about doing OT on G3's together?
## seems somewhat reasonable, actually, but what about a G3 -> G4, something on the cusp? The biology would be expected to fall between...