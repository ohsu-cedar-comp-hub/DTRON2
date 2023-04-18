from skimage.io import imread
import tifffile
import zarr
import dask.array as da
from dask import delayed
import json

def lazy_read(filenames):
	#note that files can be multichanneled, hence 'filenames'
	#import pdb;pdb.set_trace()
	sample = imread(filenames[0]) #read the first file to get the shape and dtype, which
	#we ASSUME THAT ALL OTHER FILES SHARE THE SAME SHAPE/TYPE
	lazy_imread = delayed(imread)
	lazy_arrays = [lazy_imread(fn) for fn in filenames]
	dask_arrays = [
		da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
		for delayed_reader in lazy_arrays
	]
	#Stack into one large dask.array.
	stack = da.stack(dask_arrays,axis=0)
	#stack.shape = (nfiles, nz, ny, nx)

	## For Multichannel data, we'll do something like this:
	# file_pattern = "/path/to/experiment/*ch{}*.tif"
	# channels = [imread(file_pattern.format(i)) for i in range(nchannels)]
	# stack = da.stack(channels)
	# stack.shape  # (2, 600, 64, 256, 280)

	return stack

def might_read(filenames):
	sample = imread(filenames[0])
	print(sample.dtype)
	return sample

def zarr_read(filename):
	sample = tifffile.imread(filename, aszarr=True)
	z = zarr.open(sample, mode='r')
	print(z.info)
	print(z[0].info)
	dask_arrays = [da.from_zarr(z[int(dataset['path'])]) for dataset in z.attrs['multiscales'][0]['datasets']]
	print(dask_arrays[0].dtype)	
	return dask_arrays

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data
