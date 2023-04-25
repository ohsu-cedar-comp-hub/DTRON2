"""
#############################################################
AUTHOR: CHRISTOPHER EDDY (eddy6081@gmail.com)
DATE: 10/27/20, Updated for CycIF 04/11/23
PURPOSE: SAVE Raw tiff files into a single OME.TIF file.
########################################################
"""

"""
MWE:
MM = Make_OME_TIFS(image_dir = "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/17633-scene0-3/17633-scene0-3_tiffs",
                   save_dir = "/home/exacloud/gscratch/CEDAR/cIFimaging/Cyclic_Workflow/2021_DTRON_Neuron/OME_TIF_FILES")
series_names, image_names = MM.read_filenames(image_pattern=".tif")
image_names = MM.reorder_cyc_IF_files(image_names)

### USE ONE OF THE FOLLOWING DEPENDING ON YOUR NEEDS

#### For simple applications without the need of metadata
MM.run_simple_ometif_save(series_names, image_names) #does not save any metadata, does not do tiling, etc. 

#### FOR CYC-IF DATA #####
## run this for saving multichannel cyclic IF data - will load each file into memory and hold it until complete.
MM.run_bigtif_ometif_save(series_names,image_names) # see optional arguments to include resolution and channel names!
## run this for saving multichannel IF data - will load each file as a zarr object, calling it when necessary. Particularly useful if microscope saves tiles as default in .tifs.
MM.run_bigtif_ometif_save_zarr(series_names,image_names) # see optional arguments to include resolution and channel names!

##### FOR H&E images #####
MM.run_bigtif_ometif_HandE_save(series_names, image_names, resolutionunit=None, use_compression=False) # see optional arguments
"""

###GENERATE TRAINING GRAY FILES
import tifffile
import numpy as np
import os, sys
from tifffile import TiffWriter
import zarr

class Make_OME_TIFS(object):

    def __init__(self, image_dir, save_dir=None):
        """
        INPUTS
        -----------------------------------------------------------------------
        image_dir = string, Directory of the dataset
                    "/path/to/raw_gray_tiff_files"
        save_dir = optional string, Path to save files (default=/ometifs/)
                    "/path/to/save/output.ome.tiff_files"
        """
        self.image_dir = image_dir
        if self.image_dir[-1]=="/":
            self.image_dir = self.image_dir[:-1]

        if save_dir is not None:
            if save_dir[-1]=="/":
                save_dir = save_dir[:-1]
            self.save_dir = save_dir
        else:
            self.save_dir = os.path.join(os.path.dirname(self.image_dir),"ometifs") #this goes up one directory.

        self.basename = os.path.basename(os.path.dirname(self.image_dir)) #needed for saving

        if not os.path.isdir(self.image_dir):
            raise ImportError("'image_dir' argument does not point to an exisiting directory.")

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def read_filenames(self, image_pattern = None, series_pattern = None):
        #image_pattern could be "_ch02.tif"
        if image_pattern is not None:
            assert isinstance(image_pattern, str), "'image_pattern' argument must be type string."

        #series_pattern: if the name looks like Exp020124_Series001_t01_z18.tif for example,
        #than the series pattern should be _t, since we want the stuff BEFORE the important
        #indicators. Use "_t"

        #In the case of Cyclic IF images, which look something like "<big name stuff> Export-13_c1_ORG.tif"
        # so, we want to sort the all the filenames by this 'Export-<NN>' key. However, while the round number <NN>
        #does count as our series, we actually want to combine ALL the images together, into one complete OME.TIF. Therefore,
        #there should be no series pattern.

        if series_pattern is not None:
            assert isinstance(series_pattern, str), "'series_pattern' argument must be type string."
        im_names = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir,f))]
        if image_pattern is not None:
            im_names=[f for f in im_names if image_pattern in f] #filter for fluorescence channel
        im_names.sort()
        #it won't be enough in Cyclic IF to just sort like this. Instead, we want to sort by the export number (round)
        #then sort by the indicator "_c1_" after it. I wrote a function "reorder_cyc_IF_files" specific for that.

        #we also need the unique series numbers.
        if series_pattern is not None:
            unique_series_names = list(set([name[0:name.rfind(series_pattern)] for name in im_names]))
            #unique_series_name = [name for name in unique_series_name if "Merging_Crop" in name]
            unique_series_names.sort()
        else:
            unique_series_names = [os.path.basename(self.image_dir)]

        return unique_series_names, im_names

    def reorder_cyc_IF_files(self, im_names):
        """
        THIS IS SPECIFIC FOR OUR FILENAME STRUCTURE AT CEDAR OHSU
        """
        #im_names should be a list of strings containing filepaths.
        inds = np.array([(int(x[x.find("Export-"):].split("_")[0][-2:]), int(x[x.find("Export-"):].split("_")[1][1:])) for x in im_names]) #list of tuple containing (# round, # marker))
        sort_inds = np.lexsort((inds[:,1], inds[:,0])) #works, I checked.
        im_names = [im_names[i] for i in sort_inds]
        return im_names

    def run_simple_ometif_save(self, unique_series_name, im_names):
        #print("WARNING: You may need to edit code of run_ometif_save for your file naming convention.")
        #assert self.check, "First, it is advised you check that the naming convention is correct before proceeding by running 'check_grouping', or set self.check to True."
        #print("First, it is advised you check ")
        for num,series in enumerate(unique_series_name):
            progbar(num,len(unique_series_name),20)

            images = [tifffile.imread(os.path.join(self.image_dir,image_name)) for image_name in im_names]
            images = np.stack(images)

            tifffile.imwrite(os.path.join(self.save_dir,series+".ome.tif").replace(" ", "_"),images)
            #above, replace spaces with underscores.

        progbar(len(unique_series_name),len(unique_series_name),20)
        print("\n")
        print("Complete :) \n")

    #channel names can be determined from Ece or filenames, but could be passed here!
    #@profile
    def run_bigtif_ometif_save(self, unique_series_name, im_names, channel_names = None, pixelsize = None, resolutionunit='CENTIMETER', subresolutions = 2, tilesize=512, use_compression = False):
        """
        pixelsize = float, microns / pixel
        subresolution = int, number of magnifications to compress image for pyramid structure
        tilesize = int, size of tiles in pixels
        use_compression = 'jpeg' or None or False
        """
        #init_util = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        assert resolutionunit in ["MILLIMETER", "CENTIMETER", None], "argument 'resolutionunit' was {}, but must be in {}".format(resolutionunit, ["MILLIMETER", "CENTIMETER"])
          
        if use_compression:
            use_compression = 'jpeg'

        if resolutionunit=="MILLIMETER":
            conversion = 1e3 #microns/millimeter 
        elif resolutionunit=="CENTIMETER":
            conversion = 1e4 # microns/millimeter?

        if pixelsize is None:
            pixelsize = 1 #throw a default option here.
            conversion = 1.
            
        for num,series in enumerate(unique_series_name):
            progbar(num,len(unique_series_name),20)

            images = [tifffile.imread(os.path.join(self.image_dir,image_name)) for image_name in im_names]
            images = np.stack(images)
            #now the images should be C x Y x X.
            assert len(images.shape)==3, "'images' has shape {}, needs to be shape (Channels x X x Y), or you need to adjust function run_bigtif_ometif_save".format(images.shape)

            if channel_names is None:
                channel_names = ["Channel {}".format(i) for i in range(len(im_names))]


            with TiffWriter(os.path.join(self.save_dir,series+".ome.tif").replace(" ", "_"), bigtiff=True) as tif:

                options = dict(
                    photometric='minisblack',
                    tile=(tilesize, tilesize),
                    compression = use_compression,
                    resolutionunit='CENTIMETER'
                )

                if pixelsize is not None:
                    metadata={
                        'axes': 'CYX',
                        'SignificantBits': 10,
                        'PhysicalSizeX': pixelsize,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': pixelsize,
                        'PhysicalSizeYUnit': 'µm',
                        'Channel': {'Name': channel_names},
                        'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }

                else:
                    metadata={
                    'axes': 'CYX',
                    'SignificantBits': 10,
                    'PhysicalSizeY': pixelsize,
                    'PhysicalSizeX': pixelsize,
                    'Channel': {'Name': channel_names},
                    'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }
                #self.checking_in()
                tif.write(
                        images,
                        subifds=subresolutions,
                        resolution=(conversion / pixelsize, conversion / pixelsize),
                        metadata=metadata,
                        **options
                )

                #self.checking_in()
                # write pyramid levels to the two subifds
                # in production use resampling to generate sub-resolution images
                for level in range(subresolutions):
                    mag = 2**(level + 1)
                    tif.write(
                        images[:, ::mag, ::mag],
                        subfiletype=1,
                        resolution=(conversion / mag / pixelsize, conversion / mag / pixelsize),
                        **options
                    )
                
                #self.checking_in()
                # add a thumbnail image as a separate series
                # it is recognized by QuPath as an associated image
                thumbnail = (images[0, ::16, ::16] >> 2)#.astype('uint8')
                tif.write(thumbnail, metadata={'Name': 'thumbnail'})

            tif.close()

        #final_util = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        #print(final_util - init_util)
        progbar(len(unique_series_name),len(unique_series_name),20)
        print("\n")
        print("Complete :) \n")


    def run_bigtif_ometif_HandE_save(self, unique_series_name, im_names, pixelsize = None, resolutionunit='CENTIMETER', subresolutions = 2, tilesize=512, use_compression = None):
        """
        pixelsize = float, microns / pixel
        subresolution = int, number of magnifications to compress image for pyramid structure
        tilesize = int, size of tiles in pixels
        use_compression = 'jpeg' or None or False

        resolutionunit can be None type, but the default saving behavior of tifffile is to put the ometif into tifffile format, which defaults the resolutionunit to inch. 
        """
        assert resolutionunit in ["MILLIMETER", "CENTIMETER", None], "argument 'resolutionunit' was {}, but must be in {}".format(resolutionunit, ["MILLIMETER", "CENTIMETER"])
        
        if use_compression:
            use_compression = 'jpeg'
        
        if resolutionunit=="MILLIMETER":
            conversion = 1e3 #microns/millimeter 
        elif resolutionunit=="CENTIMETER":
            conversion = 1e4 # microns/millimeter?

        for num,series in enumerate(unique_series_name):
            progbar(num,len(unique_series_name),20)

            images = [tifffile.imread(os.path.join(self.image_dir,image_name)) for image_name in im_names]
            images = np.stack(images)
            #squeeze the image.
            images = np.squeeze(images) #shape is X x Y x C

            #now the images should be X x Y x C.
            assert len(images.shape)==3, "'images' has shape {}, needs to be shape (Channels x X x Y), or you need to adjust function run_bigtif_ometif_save".format(images.shape)

            channel_names = ['Red','Green','Blue']

            if pixelsize is None:
                pixelsize = 1 #throw a default option here.
                conversion = 1.

            with TiffWriter(os.path.join(self.save_dir,series+".ome.tif").replace(" ", "_"), bigtiff=True) as tif:
                
                if pixelsize is not None:
                    metadata={
                        'axes': 'YXS',
                        'SignificantBits': 10,
                        'PhysicalSizeX': pixelsize,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': pixelsize,
                        'PhysicalSizeYUnit': 'µm',
                        'Channel': {'Name': channel_names},
                        'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }
                    options = dict(
                        photometric='rgb',
                        tile=(tilesize, tilesize),
                        compression = use_compression,
                        resolutionunit=resolutionunit,
                    )

                else:
                    metadata={
                        'axes': 'YXS', 
                        'SignificantBits': 10,
                        'PhysicalSizeX': pixelsize,
                        'PhysicalSizeY': pixelsize,
                        'Channel': {'Name': channel_names},
                        'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }
                    options = dict(
                        photometric='rgb',
                        tile=(tilesize, tilesize),
                        compression = use_compression,
                        resolutionunit=resolutionunit,
                    )


                tif.write(
                        images,
                        subifds=subresolutions,
                        resolution=(conversion / pixelsize, conversion / pixelsize), 
                        metadata=metadata,
                        **options
                )

                # write pyramid levels to the two subifds
                # in production use resampling to generate sub-resolution images
                for level in range(subresolutions):
                    mag = 2**(level + 1)
                    tif.write(
                        images[::mag, ::mag, :],
                        subfiletype=1,
                        resolution=((conversion / mag) / pixelsize, (conversion / mag) / pixelsize),
                        **options
                    )
                #the example on tiffile has subfiletype = 1, but according to the documentation, if it is just a subresolution image then it should be subfiletype=0
                # add a thumbnail image as a separate series
                # it is recognized by QuPath as an associated image
                thumbnail = (images[::16, ::16, :] >> 2)#.astype('uint8')
                tif.write(thumbnail, metadata={'Name': 'thumbnail'})

        progbar(len(unique_series_name),len(unique_series_name),20)
        print("\n")
        print("Complete :) \n")

    #@profile
    def run_bigtif_ometif_save_zarr(self, unique_series_name, im_names, channel_names = None, pixelsize = None, resolutionunit='CENTIMETER', subresolutions = 2, tilesize=512, use_compression = False):
        """
        Purpose:
        If the images were saved with a tiling format, using Zarr can be more memory efficient to write the image. 
        
        pixelsize = float, microns / pixel
        subresolution = int, number of magnifications to compress image for pyramid structure
        tilesize = int, size of tiles in pixels
        use_compression = 'jpeg' or None or False
        """
        #init_util = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        assert resolutionunit in ["MILLIMETER", "CENTIMETER", None], "argument 'resolutionunit' was {}, but must be in {}".format(resolutionunit, ["MILLIMETER", "CENTIMETER"])
          
        if use_compression:
            use_compression = 'jpeg'

        if resolutionunit=="MILLIMETER":
            conversion = 1e3 #microns/millimeter 
        elif resolutionunit=="CENTIMETER":
            conversion = 1e4 # microns/millimeter?

        if pixelsize is None:
            pixelsize = 1 #throw a default option here.
            conversion = 1.
            
        for num,series in enumerate(unique_series_name):
            progbar(num,len(unique_series_name),20)
            
            filenames = [os.path.join(self.image_dir,image_name) for image_name in im_names]
            images_zarr = tifffile.imread(filenames, aszarr=True) 
            #convert to zarr array
            images_zarr = zarr.open(images_zarr, mode='r')
            #now the images should be C x Y x X.
            assert len(images_zarr.shape)==3, "'images' has shape {}, needs to be shape (Channels x X x Y), or you need to adjust function run_bigtif_ometif_save".format(images.shape)
            shape = images_zarr.shape
            if channel_names is None:
                channel_names = ["Channel {}".format(i) for i in range(len(im_names))]

            with TiffWriter(os.path.join(self.save_dir,series+".ome.tif").replace(" ", "_"), bigtiff=True, imagej=False) as tif:

                options = dict(
                    photometric='minisblack',
                    tile=(tilesize, tilesize),
                    compression = use_compression,
                    resolutionunit=resolutionunit
                )

                if resolutionunit is not None:
                    print("here instead")
                    metadata={
                        'PhysicalSizeX': pixelsize,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': pixelsize,
                        'PhysicalSizeYUnit': 'µm',
                        'Channel': {'Name': channel_names},
                        'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }

                else:
                    print("here")
                    metadata={
                    'Channel': {'Name': channel_names},
                    'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
                    }
                    #Create the OME metadata dictionary
                #self.checking_in()
                tif.write(
                    images_zarr,
                    subifds=subresolutions,
                    resolution=(conversion / pixelsize, conversion / pixelsize),
                    metadata=metadata,
                    **options
                )

                # write pyramid levels to the two subifds
                # in production use resampling to generate sub-resolution images
                #self.checking_in()
                
                for level in range(subresolutions):
                    mag = 2**(level + 1)
                    tif.write(
                        images_zarr[:, ::mag, ::mag],
                        subfiletype=0,
                        resolution=(conversion / mag / pixelsize, conversion / mag / pixelsize),
                        **options
                    )

                #self.checking_in()
                # add a thumbnail image as a separate series
                # it is recognized by QuPath as an associated image
                thumbnail = (images_zarr[0, ::16, ::16] >> 2)#.astype('uint8')
                tif.write(thumbnail, metadata={'Name': 'thumbnail'})

            tif.close()


        #final_util = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        #print(final_util - init_util)
        progbar(len(unique_series_name),len(unique_series_name),20)
        print("\n")
        print("Complete :) \n")

    #@profile
    def checking_in(self):
        print('yes')


    #@profile
    def checking_in(self):
        print('yes')

    

#
def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

