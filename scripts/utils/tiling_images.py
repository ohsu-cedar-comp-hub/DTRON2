"""
Author: Christopher Z. Eddy (eddyc@ohsu.edu)
Date: 04/18/23
Purpose: tile up images into a dictionary item, often useful to turn large images into more practical inputs for deep learning networks.
Restitch after detection. See minimum working example provided.
"""

import numpy as np 
import itertools

"""
MWE
#load your images, should be in format [C x X x Y]
import tifffile 
image = tifffile.imread("<your_file_name>") #works with .tif, .ome.tif
image = np.moveaxis(image,2,0)
#check the shape!
print(image.shape)
tiles = tile_image(image, tile_size = 4000, p_overlap = 0.1)
#consider saving tiles as a hdf5...
retiled_image, _ = stitch_image(tiles['tile_size'],tiles)

#tiles has a currently empty dictionary key called "mask", fill it with the masks if you are going to stitch those as well.
mask_paths = ... #provide the paths. Might want to use a find function for this, or glob
masks = [tifffile.imread(pth) for pth in mask_paths]
#check that each mask is shape (H x W), and that these files are BINARY
print(masks[0].shape)

tiles['mask'] = masks
#check the shape
retiled_image, output_mask = stitch_image(tiles['tile_size'],tiles)

#save output_mask to file.
tiffile.imwrite("<your output file>", output_mask, photometric='minisblack'))

"""
def tile_image(image, tile_size=512, p_overlap=0.15):
    """
    INPUTS
    --------------------------
    image = numpy array of shape (any, X, Y), but if multichannel must be (C x X x Y)
    tile_size = int, square tile size in pixels
    p_overlap = float (<1), overlap percentage as decimal of image tiles
    
    OUTPUTS
    --------------------------
    tiles = dictionary, 
        -keys: 
            'image': list of numpy arrays containing each tile by 'windows'
            'mask': empty list (to be filled by user)
            Not going to list all of them, but see function for details.
    """
    assert p_overlap<1, "overlap percentage (decimal) must be less than 1"
    h_im, w_im = image.shape[-2:]
    tile_size = int(tile_size)
    p_tile = int(np.ceil(tile_size*(1-p_overlap)))
    n_slices_vert = int(np.ceil(h_im / p_tile))
    n_slices_horz = int(np.ceil(w_im / p_tile))

    #Decide padding, if necessary.
    if h_im < n_slices_horz * p_tile:
        n_pad_rows = n_slices_vert*p_tile - h_im
    else:
        n_pad_rows = 0
    if w_im < n_slices_horz * p_tile:
        n_pad_cols = n_slices_horz*p_tile - w_im
    else:
        n_pad_cols = 0

    #determine start points for tiles to run detection on.
    #if doing padding, the next commented line works.
    #locs = [((ri,ri+tile_size),(ci,ci+tile_size)) for ri in range(0,h_im,p_tile) for ci in range(0, w_im, p_tile)]
    #the following code is for non-padded.
    locs=[]
    for ri in range(0,h_im,p_tile):
        for ci in range(0,w_im,p_tile):
            if ri+tile_size>h_im and ci+tile_size>w_im: #we are at the sets pf rows and columns
                locs.append(((h_im-tile_size,h_im),(w_im-tile_size,w_im)))
            elif ri+tile_size>h_im and ci+tile_size<=w_im: #we are at the last set of rows
                locs.append(((h_im-tile_size,h_im),(ci,ci+tile_size)))
            elif ri+tile_size<=h_im and ci+tile_size>w_im: #we are at the last set of columns
                locs.append(((ri,ri+tile_size),(w_im-tile_size,w_im)))
            else: #
                locs.append(((ri,ri+tile_size),(ci,ci+tile_size)))
    #There is overlap in these tiles. In CellPose, there is also edge effects when calculating
    #the gradients. Therefore, we want to only take the inner portions of each image.
    #Determine the acceptable windows of each tile image to use.
    trim_amount = int(p_overlap * tile_size/2)#int((1-p_overlap)*tile_size/2)
    hmins = [0 if ri==0 else ri + trim_amount if ri + trim_amount <= h_im else ind*tile_size for ind,ri in enumerate(range(0,h_im,p_tile))]#for ((ri,_),(_,_)) in locs]
    hmaxs = [ri + tile_size - trim_amount if ri + tile_size < h_im else h_im for ri in range(0,h_im,p_tile)]
    wmins = [0 if wi==0 else wi + trim_amount if wi + trim_amount <= w_im else ind*tile_size for ind,wi in enumerate(range(0,w_im,p_tile))]#for ((ri,_),(_,_)) in locs]
    wmaxs = [wi + tile_size - trim_amount if wi + tile_size < w_im else w_im for wi in range(0, w_im, p_tile)]

    #locs looks like list of each tile's (initial row, final row), (initial column, final column)

    rows = list(zip(hmins,hmaxs))
    cols = list(zip(wmins,wmaxs))
    windows = list(itertools.product(rows,cols))
    windows = [tuple(x[0] + x[1]) for x in windows]
    #create tiles dictionary
    tiles = {}
    #tiles['locs']=locs
    tiles['padding']=[n_pad_rows,n_pad_cols]
    tiles['init_shape']=image.shape
    tiles['channels'] = image.shape[0] if len(image.shape) > 2 else 1#self.config.OUT_CHANNELS
    tiles['dtype']=image.dtype
    tiles['trim']=trim_amount
    tiles['windows']=windows
    tiles['image']=[image[..., x[0][0]:x[0][1],x[1][0]:x[1][1]] for x in locs]#im_tiles#[]
    tiles['mask']=[]
    tiles['locs'] = locs
    tiles['tile_size'] = tiles['image'][0].shape[-1] #images should be square.

    #verify locs and windows line up. Checked, looks good.
    return tiles

def stitch_image(tile_size, tiles):
    """
    INPUTS
    ---------------------------------
    tile_size = int, or tiles['tile_size'] corresponding to image tile sizes
    tiles = dictionary, with args 'windows', 'dtype', 'image', 'padding', 'init_shape', 'trim', and 'mask'
        each key should be a list with tiles.
    
    OUTPUTS
    ---------------------------------
    image = numpy array, restitched image
    mask = numpy array or empty list, restitched mask array or empty list if tiles['mask'] is an empty list.
    """

    #for image, stitch together using max projection
    shape_to=tiles['init_shape']#[int(np.sum(el)) for el in zip(list(tiles['init_shape']),tiles['padding'])]
    image = np.zeros(shape=tuple(shape_to), dtype = tiles['dtype'])

    for tile_i in range(len(tiles['image'])):
        (ri,rend,ci,cend) = tiles['windows'][tile_i] #these are placements into image.

        if ri==0:
            rstart = 0
        elif rend==tiles['init_shape'][-2]:
            rstart = tile_size-(rend-ri)
        else:
            rstart = tiles['trim']
        if ci==0:
            cstart = 0
        elif cend==tiles['init_shape'][-1]:
            cstart=tile_size-(cend-ci)
        else:
            cstart = tiles['trim']
        if rend < tiles['init_shape'][-2]:
            rstop = tile_size - tiles['trim']
        else:
            rstop = tile_size
        if cend < tiles['init_shape'][-1]:
            cstop = tile_size - tiles['trim']
        else:
            cstop = tile_size
        #so to be clear, rstart,rstop,cstart,cstop are the indices for the tile.
        this_im = tiles['image'][tile_i][...,rstart:rstop,cstart:cstop] #note the 0 is just since there was an added channel.

        image[...,ri:rend,ci:cend] = this_im


    if len(tiles['mask'])>0:
        #for mask, remove objects that are on the border. For deleted pixels, delete the same Pixels
        mask = np.zeros(shape=tuple(shape_to[-2:]), dtype = tiles['mask'][0].dtype)
        for tile_i in range(len(tiles['image'])):

            (ri,rend,ci,cend) = tiles['windows'][tile_i] #these are placements into image.

            if ri==0:
                rstart = 0
            elif rend==tiles['init_shape'][-2]:
                rstart = tile_size-(rend-ri)
            else:
                rstart = tiles['trim']
            if ci==0:
                cstart = 0
            elif cend==tiles['init_shape'][-1]:
                cstart=tile_size-(cend-ci)
            else:
                cstart = tiles['trim']
            if rend < tiles['init_shape'][-2]:
                rstop = tile_size - tiles['trim']
            else:
                rstop = tile_size
            if cend < tiles['init_shape'][-1]:
                cstop = tile_size - tiles['trim']
            else:
                cstop = tile_size

            this_mask = tiles['mask'][tile_i][rstart:rstop,cstart:cstop]
            
            mask[ri:rend,ci:cend] = np.where(this_mask > mask[ri:rend,ci:cend], this_mask, mask[ri:rend,ci:cend])

        image = image[...,0:tiles['init_shape'][0],0:tiles['init_shape'][1]]
        mask = mask[0:tiles['init_shape'][0],0:tiles['init_shape'][1]]
    
    else:
        mask = []

    return image, mask