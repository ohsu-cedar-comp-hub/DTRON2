from skimage.io import imread
import tifffile
import zarr
import dask.array as da
from dask import delayed
import json
from xml.etree import ElementTree

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


def zarr_read(filename, split_channels = False):
	"""
	split_channels = boolean, True if you want to display multichanneled images seperately rather than together.
	Use for cycIF or H&E where we don't care to see channel names -- i.e. not split.
	"""
	#grab the channel name metadata
	if split_channels:
		try:
			with tifffile.TiffFile(filename) as tif:
				metadta = ElementTree.fromstring(tif.series[0].pages[0].description)
			metadta = metadta[0][0]
			channel_names = [(x.attrib)['Name'] for x in metadta if len(x)>0]
		except:
			with tifffile.TiffFile(filename) as tif:
				root = tif.series[0].pages[0].description
				#parse root for SizeC
				start = root.find('SizeC=')+6
				if start!=-1:
					#that is, if it found SizeC in the string.
					subroot = root[start:]
					subroot = subroot[:subroot.find(" ")][1:-1] #find the first space, get rid of the quotation marks
					num_channels = int(subroot)
					channel_names = ["Channel {}".format(i+1) for i in range(num_channels)]
				else:
					channel_names = []

	else:
		channel_names = []

	sample = tifffile.imread(filename, aszarr=True)
	z = zarr.open(sample, mode='r')
	dask_arrays = [da.from_zarr(z[int(dataset['path'])]) for dataset in z.attrs['multiscales'][0]['datasets']]
	#added CE 042423 :: want to display multiple channels and selectively turn on or off each.
	if split_channels:
		if len(dask_arrays[0].shape)>2:
			dask_arrays = [[x[ch_i, ...] for x in dask_arrays] for ch_i in range(dask_arrays[0].shape[0]) ] #list of lists to be split.
	
	return dask_arrays, channel_names


def load_json_data(pth):
	with open(pth) as f:
		data=json.load(f)
	return data
