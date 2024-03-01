"""
AUTHOR: Christopher Z. Eddy (eddyc@ohsu.edu)
Purpose: Functions to read and process single cell average marker intensity profiles. Specifically, this set of functions are meant to normalize marker intensity features.
Notes: Erik Burlingame's implementation (included in read_file_EB) used clipping rather than removal of outlier rows, artificially saturating particular marker values. In my version, read_file_CE,
    we perform additional normalization steps as well as outlier removal, where outliers are measured for each marker independently. 
"""

import os
import pandas as pd
import numpy as np
import cuml
import cudf
import time
from sklearn.preprocessing import minmax_scale, robust_scale 
import matplotlib.pyplot as plt

def read_file(file_pth='/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/Features/Normalized Mean Intensities.csv', raw = False, return_data = True):
    start = time.time()
    source_df = cudf.read_csv(file_pth)

    if not raw:

        # Regarding the data, remove "label" = cell number 
        # columns that star with Q are quench rounds, so they are not real markers and exclude those.
        # to start, include centroid. If they group only based on spatial locality, we'll exclude it later. 
        # Remove minx, maxx, etc. 
        # Actual data starts with first column that says "EPCAM"
        # Q4 starts after NOTCH1.
        # Basically remove any columns starting with Q. 
        # WANT
        # columns 5,6, 40 - 103, 120 - 215

        good_cols = source_df.columns[ list(np.arange(40,103 + 1)) + list(np.arange(120,215 + 1))]

        data = source_df[good_cols].to_pandas()

        # # remove outliers ACCORDING TO ERIC. THIS DOES NOT ACTUALLY REMOVE OUTLIERS! CE
        # data = data.clip(upper=data.quantile(q=.999),axis=1) #maybe use a lower q val to exclude higher proportion of outliers? (org is q=.999)

        #we actually just want to remove rows that are outside of the 99.9th percentile. This is what they do in phenograph.
        #I would prefer not to remove those points, but the issue is with the KNN implementation, 
        data = data.apply(robust_scale)

        ## Remove outliers?
        # from scipy import stats
        # import numpy as np
        # data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

        # deskew
        data = (data/5).apply(np.arcsinh)

        ## scale
        data = data.apply(minmax_scale)

        # # Consider log transforming the data, if the distributions appear log-normal.
        # data += 1
        # data.apply(np.log)

        ## Let's also add on the "Sample" column
        data['Sample'] = source_df['Sample'].to_pandas()
        
        #add position
        data[['CentroidX','CentroidY']] = source_df[['CentroidX','CentroidY']].to_pandas()
        #pd.Series([str(x) for x in source_df['Sample'].to_array()], dtype="category")

        #write to a file
        print("Writing to file : \n{}".format(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+"_processed.csv")))
        data.to_csv(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+"_processed.csv"),index=False)
        print("Data shape = {}".format(data.shape))
        if return_data:
            print("Converting to cudf")
            #data back to cudf
            data = cudf.from_pandas(data)
                                                                        
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))

            return data
        else:
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))
            print("Complete with preprocessing.")
    else:
        return source_df


def read_file_CE(file_pth='/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/Features/Normalized Mean Intensities.csv', raw = False, write_out_data = True, custom_ext:str = "_processed_robust_CE", return_data = True, bad_cols: list = []): #, neuronal=True):
    start = time.time()
    #good_cols = source_df.columns[ [5,6] + list(np.arange(40,103 + 1)) + list(np.arange(120,215 + 1))]
    source_df = cudf.read_csv(file_pth)

    #replace an <NA> values with np.nan
    # source_df.fillna(value=np.nan, inplace=True)

    if not raw:
        #any that have "To Exclude" as "x", should be removed. "o" should be kept.
        if "To Exclude" in source_df.columns:
            source_df = source_df.loc[~source_df['To Exclude'].isin(["x"])]
        if "To.Exclude" in source_df.columns:
            source_df = source_df.loc[~source_df['To.Exclude'].isin(["x"])]

        # Regarding the data, remove "label" = cell number 
        # columns that star with Q are quench rounds, so they are not real markers and exclude those.
        # to start, include centroid. If they group only based on spatial locality, we'll exclude it later. 
        # Remove minx, maxx, etc. 
        # Actual data starts with first column that says "EPCAM"
        # Q4 starts after NOTCH1.
        # Basically remove any columns starting with Q. 
        # WANT
        # columns 5,6, 40 - 103, 120 - 215
        # if neuronal:
        #     #good_cols = source_df.columns[ list(np.arange(40,103 + 1)) + list(np.arange(120,215 + 1))] #why is this list, the list? Check with Ece. 
        #     drop_list = ["Sample", "Label", "MinX", "MinY",	"MaxX",	"MaxY",	"CentroidX", "CentroidY", "To Exclude", "Unnamed"]
        #     good_cols = [x for x in source_df.columns.tolist() if np.all([y not in x for y in drop_list])]
        
        #drop_list = ["Sample", "Label", "MinX", "MinY",	"MaxX",	"MaxY",	"CentroidX", "CentroidY", "To Exclude", "Unnamed"]
        #drop_list should be any columns that contain strings. So grab the first row, figure out which are strings.
        drop_list = [source_df.columns[i] for i,x in enumerate(source_df.iloc[0].to_pandas().values.flatten()) if not is_float(str(x))] + [x for x in bad_cols if x in source_df.columns] #the str method is used to deal with the nan's and Nones
        drop_list = list(set(drop_list)) #the two lists may repeat one of the elements, considering you can pass whatever list in bad_cols
        good_cols = [x for x in source_df.columns.tolist() if np.all([y not in x for y in drop_list])]

        data = source_df[good_cols].to_pandas()

        #if To Exclude is in drop list, drop it.
        drop_list = [x for x in drop_list if ('To Exclude' not in x) and ('To.Exclude' not in x) and (x in source_df.columns)] #we don't want to put to exlude back in the data. 

        # # remove outliers ACCORDING TO ERIC. THIS DOES NOT ACTUALLY REMOVE OUTLIERS! CE
        # data = data.clip(upper=data.quantile(q=.999),axis=1) #maybe use a lower q val to exclude higher proportion of outliers? (org is q=.999)

        #we actually just want to remove rows that are outside of the 99.9th percentile. This is what they do in phenograph.
        q_dta = data.quantile(q=0.999,axis=0)
        for (col, val) in zip(q_dta.index.to_list(),list(q_dta.values)):
            data=data.loc[data[col]<=val]
            
        #implement robust scaling 
        data = data.apply(robust_scale)

        # deskew
        data = (data/5).apply(np.arcsinh) #while this is primarily linear in the range of -1 to 1, beyond 1 and -1 it significantly plateus.

        ## scale
        data = data.apply(minmax_scale)

        # # Consider log transforming the data, if the distributions appear log-normal.
        # data += 1
        # data.apply(np.log)

        ## Let's also add on the "Sample" column
        # data['Sample'] = source_df['Sample'].to_pandas()
        # #add position
        # data[['CentroidX','CentroidY']] = source_df[['CentroidX','CentroidY']].to_pandas()
        # #pd.Series([str(x) for x in source_df['Sample'].to_array()], dtype="category")
        """ Adding back the string columns """
        data[drop_list] = source_df.loc[data.index][drop_list].to_pandas()

        #write to a file
        if write_out_data:
            print("Writing to file : \n{}".format(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+custom_ext +".csv")))
            data.to_csv(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+custom_ext),index=False)
            
        print("Data shape = {}".format(data.shape))
        
        if return_data:
            print("Converting to cudf")
            #data back to cudf
            data = cudf.from_pandas(data)
                                                                        
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))

            return data, drop_list
        else:
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))
            print("Complete with preprocessing.")
    else:
        return source_df

def read_file_EB(file_pth='/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/Features/Normalized Mean Intensities.csv', raw = False, return_data = True):
    """
    Load and process the file using Eric Burlingame's processing, per Ece's request.
    Again, this processing is not correct.
    """
    start = time.time()
    #good_cols = source_df.columns[ [5,6] + list(np.arange(40,103 + 1)) + list(np.arange(120,215 + 1))]
    source_df = cudf.read_csv(file_pth)

    if not raw:

        # Regarding the data, remove "label" = cell number 
        # columns that star with Q are quench rounds, so they are not real markers and exclude those.
        # to start, include centroid. If they group only based on spatial locality, we'll exclude it later. 
        # Remove minx, maxx, etc. 
        # Actual data starts with first column that says "EPCAM"
        # Q4 starts after NOTCH1.
        # Basically remove any columns starting with Q. 
        # WANT
        # columns 5,6, 40 - 103, 120 - 215

        good_cols = source_df.columns[ list(np.arange(40,103 + 1)) + list(np.arange(120,215 + 1))]

        data = source_df[good_cols].to_pandas()

        # # remove outliers ACCORDING TO ERIC. THIS DOES NOT ACTUALLY REMOVE OUTLIERS! CE
        data = data.clip(upper=data.quantile(q=.999),axis=1) #maybe use a lower q val to exclude higher proportion of outliers? (org is q=.999)

        # deskew
        data = (data/5).apply(np.arcsinh)

        ## scale
        data = data.apply(minmax_scale)

        ## Let's also add on the "Sample" column
        data['Sample'] = source_df['Sample'].to_pandas()
        #pd.Series([str(x) for x in source_df['Sample'].to_array()], dtype="category")

        #write to a file
        print("Writing to file : \n{}".format(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+"_processed_EB.csv")))
        data.to_csv(os.path.join(os.path.dirname(file_pth), os.path.basename(file_pth)[:-4]+"_processed_EB.csv"),index=False)
        print("Data shape = {}".format(data.shape))
        if return_data:
            print("Converting to cudf")
            #data back to cudf
            data = cudf.from_pandas(data)
                                                                        
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))

            return data
        else:
            end = time.time()

            print("total preprocessing time = {} minutes".format((end-start)/60))
            print("Complete with preprocessing.")
    else:
        return source_df

def load_data(file_pth):
    data = cudf.read_csv(file_pth)
    return data

def is_float(string):
	if string is not None:
		if string.replace(".", "").replace("-","").isnumeric():
			return True
		else:
			return False
	else:
		return False

"""
Some notes:
the returned dataframe has an extra column ('Sample') that was not previously included. We will 
want to exclude it PRIOR to doing UMAP embedding.
Just use this:
df.drop(['Sample'], axis=1) # note that in_place is not there, so it just temporarily does the operation.
"""

def visualize_hists(data, col_name=None, save_fig=False, figname=None, plot_mean=False):
    exclude_list = ['Sample','Label', 'MinX', 'MinY', 'MaxX', 'MaxY', 'CentroidX', 'CentroidY','To Exclude']
    exclude_list = [x for x in exclude_list if x in data.columns.to_list()]
    if figname is None:
        figname = "figure.pdf"
        
    dta = data.to_pandas()

    if col_name is not None:
        ax = dta.hist(col_name,bins=100)
        #get means
        if plot_mean:
            for r in range(ax.shape[0]):
                for c in range(ax.shape[1]):
                    ttl = ax[r,c].get_title()
                    if ttl!="":
                        ax[r,c].axvline(dta[ttl].mean(), color='k', linestyle='dashed', linewidth=2)
        if save_fig:
            fig = plt.gcf()
            fig.savefig(figname)
    else:
        ax = dta.drop(exclude_list,axis=1).hist(bins=30, figsize=(45, 30))
        #get means
        if plot_mean:
            for r in range(ax.shape[0]):
                for c in range(ax.shape[1]):
                    ttl = ax[r,c].get_title()
                    if ttl!="":
                        ax[r,c].axvline(dta[ttl].mean(), color='k', linestyle='dashed', linewidth=2)
        if save_fig:
            #ax.get_figure().savefig(figname+'.pdf')
            #ax.figure.savefig(figname, dpi=500)
            fig = plt.gcf()
            fig.savefig(figname)