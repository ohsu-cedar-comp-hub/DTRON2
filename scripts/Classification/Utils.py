"""
Author: Christopher Z. Eddy
Date: 02/07/23
Purpose:
Any secondary functions useful to training or inference should go here.
"""

import os
import glob 

def import_filenames(filepath, pattern, recursive = False):
		"""
		PURPOSE: Imports ALL files from a chosen folder
		INPUTS
		-------------------------------------------------------------
		pattern = list of strings with particular patterns, including filetype! 
				ex: ["_patched",".csv"] will pull any csv files under filepath with the string "_patched" in its file name.

		filepath = string, path for where to search for image files 
				ex: "/users/<username>/folder"

		recursive = boolean, True if you wish for the search for files to be recursive under filepath.
		"""
		#generate pattern finding 
		fpattern = ["**{}".format(x) for i,x in enumerate(pattern)]
		fpattern = "".join(fpattern)
		#first, get all the filenames in the chosen directory.
		#root directory needs a trailing slash
		if filepath[-1]!="/":
				filepath = filepath+"/"
		fnames = [filename for filename in glob.iglob(filepath + fpattern, recursive=recursive)] #best for recursive search one liner.
		fnames = [x for x in fnames if x[0]!="."] #delete hidden files
		fnames.sort() #sort based on name
		return fnames

def load_config_file(fpath, config):
	"""
	Replace previous config attributes with a previously saved models attributes.
	See Config.py 'write_to_txt' function
	"""
	if not os.path.isfile(fpath):
		print("\n README.txt config file does not exist in weights path. Proceeding with default configuration...\n")
	else:
		import ast
		import re
		README_dict = {}
		with open(fpath) as file:
			lines = [line.rstrip() for line in file]
		begin=False

		for i,line in enumerate(lines):
			if line=='CONFIGURATION SETTINGS:':
				#want to begin on the NEXT line.
				begin=True
				continue

			if begin==1:
				(key, val) = line.split(maxsplit=1)
				try:
					#anything that is not MEANT to be a string.
					#mostly this does fine on its own.
					README_dict[key] = ast.literal_eval(val)
				except:
					try:
						#messes up list attributes where there spaces are not uniform sometimes.
						README_dict[key] = ast.literal_eval(re.sub("\s+", ",", val.strip()))
					except:
						README_dict[key] = val

		print("\n Replacing default config key values with previous model's config file... \n")
		for func in dir(config):
			if not func.startswith("__") and not callable(getattr(config, func)):
				#print("{:30} {}".format(a, getattr(self, a)))
				if func in README_dict.keys():
					#special case if it is a dictionary.
					if isinstance(README_dict[func],dict):
						#change keys if they exist in config.
						#get the dictionary from config.
						config_dict = getattr(config, func)
						#change values.
						#get keys in README_dict[func]
						for RM_key in README_dict[func].keys():
							#if RM_key in config_dict.keys():
							#	#reset value.
							config_dict[RM_key] = README_dict[func][RM_key]
							#else:
							#	#add new key...
						#set into config.
						setattr(config, func, config_dict)
					else:
						setattr(config, func, README_dict[func])

	return config