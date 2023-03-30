from skimage.io import imread
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

def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data