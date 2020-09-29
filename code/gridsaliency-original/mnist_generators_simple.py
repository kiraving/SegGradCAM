#import os
import math
import numpy as np
import hashlib
from matplotlib import pyplot as plt

#from copy import deepcopy
from os import listdir
from os.path import join
from PIL import Image
from skimage.transform import resize
from keras.datasets import mnist

mnist_random = None

"""
Prepares textures for further processing by tiling them to the specified
resolution.

Args:
    textures(list of 2D np arrays): Raw textures
    res(int): Resolution to which the textures are tiled.

Returns:
    Tiled textures
"""
def get_textures(textures, res):
    output = []
    for k in range(0,len(textures)):
        tex = textures[k]
        min_dim = np.min([tex.shape[0], tex.shape[1]])
        # Only tile if the original texture is smaller than the target
        if min_dim < res:
            tile = math.ceil(res/min_dim)
            tex = np.tile(tex,(tile,tile))
        output.append(tex / 255)
    return output

"""
Crop a random section from the texture.

Args:
    source(2D np array): Full size texture
    res(int): Resolution of the desired crop

Returns:
    Randomly cropped texture
"""
def random_crop(source,res):
    global mnist_random
    # Random crop of the sources
    assert((source.shape[0] > res) and (source.shape[1] > res))
    xstart_1 = mnist_random.randint(source.shape[0]-res)
    ystart_1 = mnist_random.randint(source.shape[1]-res)
    xstart_2 = mnist_random.randint(source.shape[0]-res)
    ystart_2 = mnist_random.randint(source.shape[1]-res)

    return source[xstart_1:xstart_1+res,ystart_1:ystart_1+res]

"""
Converts index to tile.

Args:
    i(int): Tile index
    split(int): Number of tiles in image (2 or 4)
    res(int): Resolution of the image

Returns:
    x(int): X-coordinate of tile position in the image
    y(int): Y-coordinate of tile position in the image
    tsize(int): Size of the tile
"""
def index_to_tile(i, split, res):
    assert (res % 2) == 0
    if split == 2:
        tsize = (res // 2, res)
        x = i
        y = 0
    elif split == 4:
        tsize = (res // 2, res // 2)
        x = i // 2
        y = i % 2
    else:
        raise ValueError('Invalid split number', split)
    return x, y, tsize

"""
Generate tiled background with randomly cropped textures

Args
    sources(list of 2D np arrays): Source textures
    res(int): Image resolution

Returns:
    agg(2D np array): Tiled background image
"""
def random_crop_agg(sources,res):
    global mnist_random
    split = len(sources)

    agg = np.zeros((res,res))
    for i,source in enumerate(sources):
        # Random crop of the sources
        x,y,tsize = index_to_tile(i, split, res)
        assert((source.shape[0] > tsize[0]) and (source.shape[1] > tsize[1]))
        xstart_1 = mnist_random.randint(source.shape[0]-tsize[0])
        ystart_1 = mnist_random.randint(source.shape[1]-tsize[1])
        xstart_2 = mnist_random.randint(source.shape[0]-tsize[0])
        ystart_2 = mnist_random.randint(source.shape[1]-tsize[1])
        agg[x*tsize[0]:(x+1)*tsize[0], y*tsize[1]:(y+1)*tsize[1]] =\
            source[xstart_1:xstart_1+tsize[0],ystart_1:ystart_1+tsize[1]]

    return agg

"""
Select combination of foreground and background textures, contingent on
relevant criteria being met.

Args:
    sources_1 (2D np arrays): List of available foreground textures
    source_labels_1 (strings): List of foreground texture labels
    sources_2 (2D np arrays): List of available background textures
    source_labels_2 (strings): List of background texture labels
    unbiased_sources_2 (2D np arrays): List of unbiased textures
    unbiased_source_labels_2 (strings): List of unbiased texture labels
    excluded_ids (ints): List of indexes of textures that are excluded.
    source2_split (int): Number of tiles the background is split into
    res(int): Image resolution

Returns:
    source_1_img: Foreground texture
    source_2_agg (2D np array): Background image with possibly multiple
        texture tiles
    source_1_label (string): Label of chosen foreground texture
    source_2_labels (strings): Label(s) of chosen background textures
"""
def sel_paired_textures(sources_1, source_labels_1,
                        sources_2, source_labels_2,
                        unbiased_sources_2, unbiased_source_labels_2,
                        excluded_ids,
                        source2_split, res):
    global mnist_random
    if source2_split == 1:
        while True:
            source_1_id = mnist_random.randint(0,len(sources_1))
            source_2_id = mnist_random.randint(0,len(sources_2))
            source_1 = sources_1[source_1_id]
            source_2 = sources_2[source_2_id]
            source_1_label = source_labels_1[source_1_id]
            source_2_label = source_labels_2[source_2_id]
            if not(source_1_id in excluded_ids or source_2_id in excluded_ids):
                break

        source_1 = random_crop(source_1,res)
        source_2 = random_crop(source_2,res)

        return source_1, source_2, source_1_label, source_2_label
    else:
        while True:
            source_1_id = mnist_random.randint(0,len(sources_1))
            source_1_img = sources_1[source_1_id]
            source_1_label = source_labels_1[source_1_id]
            if source_1_id not in excluded_ids:
                break

        source_2_ids, source_2_imgs, source_2_labels = [],[],[]
        if len(sources_2) < len(unbiased_sources_2):
            biased_tile = mnist_random.randint(0,source2_split)
        else:
            biased_tile = -1
        for i in range(source2_split):
            while True:
                if i == biased_tile:
                    source_2_id = mnist_random.randint(0,len(sources_2))
                    source_2_img = sources_2[source_2_id]
                    source_2_label = source_labels_2[source_2_id]
                else:
                    source_2_id = mnist_random.randint(0,len(unbiased_sources_2))
                    source_2_img = unbiased_sources_2[source_2_id]
                    source_2_label = unbiased_source_labels_2[source_2_id]
                if source_2_id not in excluded_ids:
                    break
            source_2_imgs.append(source_2_img)
            source_2_labels.append(source_2_label)

        source_1_img = random_crop(source_1_img,res)
        source_2_agg = random_crop_agg(source_2_imgs,res)

        return source_1_img, source_2_agg, source_1_label, source_2_labels

"""
Applies one source (texture) to a batch of targets before moving on to next
source.

Args:
    sources (2D np arrays): List of source textures in full size.
    source_labels (strings): List of string identifiers for each source,
        in same order.
    targets (2D np arrays): MNIST binary digits
    labels (np array): Digit labels
    res (int): Resolution of the output image (res x res)
    bias (dict): Dictionary specifying probabilities of combining sources and
        targets. By default, random sources are applied to each target. Here,
        this behavior can be overriden. If None, apply default behavior

        The keys of this dictionary are target labels. If a label does not appear
        as a key, the default behavior above is assumed. The value for a key
        is another dictionary with keys:
            "source_1_id"  - the label of the foreground to add a bias for
            "source_2_id"  - the label of the background to add a bias for
            "source_1_bias" - float between 0 and 1, how likely to assign
                source_1 to the foreground when this key is present.
            "source_2_bias" - float between 0 and 1, how likely to assign
                source_2 to the background when this key is present.

Returns:
    Batches of tuples (image, target, masks_dict)
"""
def gen_apply_tex_to_batch(sources, source_labels, targets, labels, res,
    bias, exclude_bias_textures,each_texture_random,background_split, return_dict=True):
    global mnist_random

    # Get indexing for individual targets (digits)
    # Randomize order of instances for each class
    labels_set = list(set(labels))
    labels_dict = {}
    for k in labels_set:
        labels_dict[k] = np.where(labels==k)[0]
        labels_dict[k] = mnist_random.permutation(labels_dict[k])

    sources_fg = sources
    source_labels_fg = source_labels
    sources_bg = sources
    source_labels_bg = source_labels
    sources = None
    source_labels = None
    # print(source_labels_fg)
    # print(source_labels_bg)

    excluded_textures = []
    for t_label in range(10):
        if (bias is not None) and (t_label in bias.keys()):
            bias_data = bias[t_label]
            if exclude_bias_textures:
                if bias_data["source_1_bias"] != 0.0:
                    source_fg_bias_idx = np.where(np.array(source_labels_fg)==bias_data["source_1_id"])[0][0]
                    excluded_textures.append(source_fg_bias_idx)
                if bias_data["source_2_bias"] != 0.0:
                    source_bg_bias_idx = np.where(np.array(source_labels_bg)==bias_data["source_2_id"])[0][0]
                    excluded_textures.append(source_bg_bias_idx)

    counter = 0
    while True:

        batch_idx = [labels_dict[k][counter % len(labels_dict[k])] \
            for k in labels_dict.keys()]

        batch_labels = [k for k in labels_dict.keys()]
        counter = counter + 1

        # Get the current source pair.
        # One source pair gets applied to all targets in the current batch
        # Choose sources randomly contingent on meeting required constraints.
        source_1_rand, source_2_rand, source_1_label, source_2_label = \
            sel_paired_textures(
                sources_fg, source_labels_fg,
                sources_bg, source_labels_bg,
                sources_bg, source_labels_bg,
                excluded_textures,
                background_split, res)

        # For each target in the current batch, apply the sources
        batch = []
        seg = []
        target_label = []
        for k in range(len(batch_idx)):
            if each_texture_random:
                source_1_rand, source_2_rand, source_1_label, source_2_label = \
                    sel_paired_textures(
                        sources_fg, source_labels_fg,
                        sources_bg, source_labels_bg,
                        sources_bg, source_labels_bg,
                        excluded_textures,
                        background_split, res)

            t_label = batch_labels[k]

            # If there is no bias, we just take the predefined texture assignment
            source_1_use = source_1_rand
            source_2_use = source_2_rand
            source_1_label_use = source_1_label
            source_2_label_use = source_2_label

            # If there is a bias, need to override the default selection
            is_biased = False
            if (bias is not None) and (t_label in bias.keys()):
                bias_data = bias[t_label]

                # Foreground
                bias_1 = bias_data["source_1_bias"]
                if mnist_random.uniform() <= bias_1:
                    is_biased = True
                    source_1_idx = np.where(np.array(source_labels_fg)==bias_data["source_1_id"])[0][0]
                    source_1_use, source_2_use, source_1_label_use, source_2_label_use = \
                    sel_paired_textures(
                        [sources_fg[source_1_idx]], [source_labels_fg[source_1_idx]],
                        sources_bg, source_labels_bg, sources_bg, source_labels_bg,
                        [], background_split, res)

                # Background
                bias_2 = bias_data["source_2_bias"]
                if mnist_random.uniform() <= bias_2:
                    is_biased = True
                    source_2_idx = np.where(np.array(source_labels_bg)==bias_data["source_2_id"])[0][0]
                    source_1_use, source_2_use, source_1_label_use, source_2_label_use = \
                    sel_paired_textures(
                        sources_fg, source_labels_fg,
                        [sources_bg[source_2_idx]], [source_labels_bg[source_2_idx]],
                        sources_bg, source_labels_bg,
                        [], background_split, res)

            # Get the current target and apply the source data
            target = resize(targets[batch_idx[k]],(res,res), order=1, anti_aliasing=False, mode='constant')
            target[target >= 0.5] = 1
            target[target < 0.5] = 0

            applied = target * source_1_use + (1-target)*source_2_use

            multi_label_target = np.zeros((*target.shape, 11))
            multi_label_target[...,t_label] = target

            background = 1 - np.sum(multi_label_target,-1)
            assert np.min(background) == 0
            assert np.max(background) == 1
            multi_label_target[...,-1] = background

            biased_tile = np.zeros((res,res))
            for j in range(background_split):
                if is_biased and bias_data["source_2_id"] == source_2_label_use[j]:
                    x,y,tsize = index_to_tile(j,background_split,res)
                    biased_tile[x*tsize[0]:(x+1)*tsize[0], y*tsize[1]:(y+1)*tsize[1]] =\
                        np.ones(tsize)

            masks_dict = {
                'digit_with_infill': np.expand_dims(target,-1),
                'background': np.expand_dims(background,-1),
                'biased_tile': np.expand_dims(biased_tile,-1),
                'is_biased': is_biased
            }
            if return_dict:
                yield(np.expand_dims(applied,-1), multi_label_target, masks_dict)
            else:
                yield(np.expand_dims(applied,-1), multi_label_target)

"""
Converts samples organized by tuples into samples organized by outputs.
The output of this function is a list with as many output variables as
provided by the generator; each output summarizes over the number of
samples produced
"""
def reduce_gen_samples(batches):

    if len(batches)==0: return None

    samples = []
    for j in range(0,len(batches[0])):
        samples.append(np.array([k[j] for k in batches]))
    return samples


"""
Generate samples from MNIST with custom foreground and
background textures.

Args:
    config: Dictionary with the following keys:
        textures_path (string): Path to directory with textures to use
        tex_res (int): The resolution of the textures to use. Must be larger than
            the resolution of the MNIST tiles.
        tile_size (int): The size of the output tiles (tile_size x tile_size)
        train_samples (int): the number of training samples to generate
        test_samples (int): The number of test samples to generate
        dataset_seed (int): Some seed integer used for setting up the random
            generator.
        fix_test_set (bool): If true, ignore dataset seed for test set.
        background_split (int): Number of tiles used for the background.
        exclude_bias_texture (bool): If true, the bias texture is excluded
            from unbiased samples.

Returns:
    Generator yielding batches (Images, Targets, Masks)
"""

def gen_texture_mnist(config, split='train', return_dict=True):
    # return_dict is a modification
    global mnist_random

    reinit_after_epoch = False
    tile_res = config["tile_size"]
    bias = config["bias"]

    if "exclude_bias_textures" in config:
        exclude_bias_textures = config["exclude_bias_textures"]
    else:
        print('exclude_bias_textures not found in config. Set to False.')
        exclude_bias_textures = False
    if "fix_test_set" in config:
        fix_test_set = config["fix_test_set"]
    else:
        print('fix_test_set not found in config. Set to False.')
        fix_test_set = False
    if "background_split" in config:
        background_split = config["background_split"]
    else:
        print('background_split not found in config. Set to 1.')
        background_split = 1

    split_int_hash = int(hashlib.md5(split.encode('utf-8')).hexdigest(),16)
    seed = (config['dataset_seed']+split_int_hash) % (2**32-1)
    if fix_test_set and split=='test':
        seed = split_int_hash % (2**32-1)
        reinit_after_epoch = True
    # print(seed)

    # Step 1 - prepare the set of textures
    texture_dir = config["textures_path"]
    tex_source = listdir(texture_dir)
    tex_base = [np.array(Image.open(join(texture_dir,k)).convert('L')) \
                for k in tex_source]
    textures = get_textures(tex_base, config["tex_res"])

    # Step 2 - prepare MNIST. Do not do rescaling yet, this will eat up
    # memory quickly
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.round(x_train/255,0)
    x_test = np.round(x_test/255,0)

    # Step 3 - generate training and testing examples in infinite loop
    while True:
        mnist_random = np.random.RandomState(seed)
        batch_size = config["batch_size"]
        if split=='train':
            num_samples = config['train_samples']
            gen = gen_apply_tex_to_batch(textures, tex_source, x_train, y_train,
                tile_res, bias, exclude_bias_textures, False,
                background_split, return_dict=return_dict)
        elif split=='test':
            num_samples = config['test_samples']
            gen = gen_apply_tex_to_batch(textures, tex_source, x_test, y_test,
                tile_res, bias, exclude_bias_textures, True,
                background_split, return_dict=return_dict)
        else:
            raise ValueError('Invalid split '+split)

        if reinit_after_epoch:
            assert num_samples % batch_size == 0
            for i in range(num_samples // batch_size):
                batch=[next(gen) for k in range(batch_size)]
                batch = reduce_gen_samples(batch)
                yield batch
        else:
            while True:
                batch=[next(gen) for k in range(batch_size)]
                batch = reduce_gen_samples(batch)
                yield batch

"""Added from the demo notebook"""


def hide_ticks(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)


def plot_samples(sample_generator, num_samples):
    i = 0
    for xs, ys, masks in sample_generator:
        for x, y, m in zip(xs, ys, masks):
            f, ax = plt.subplots(1, 4)
            ax[0].imshow(x[..., 0], cmap='gray')
            ax[0].set_title('Image')
            hide_ticks(ax[0])
            ax[1].imshow(np.argmax(y, -1), cmap='tab20', vmin=0, vmax=10)
            ax[1].set_title('Segmentation')
            hide_ticks(ax[1])
            ax[2].imshow(m['background'][..., 0], cmap='gray', vmin=0, vmax=1)
            ax[2].set_title('Background')
            hide_ticks(ax[2])
            ax[3].imshow(m['biased_tile'][..., 0], cmap='gray', vmin=0, vmax=1)
            ax[3].set_title('Biased Tile')
            hide_ticks(ax[3])
            plt.show()

            i += 1
            if i >= num_samples:
                break
        if i >= num_samples:
            break

