import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imageio import imread
from scipy.ndimage import zoom
from keras.utils import to_categorical
from .cityscape_labels import labels as LABELS


class Cityscapes:
    """store a train/test/val subset of the Cityscapes dataset in an object"""

    def __init__(self, scale=2, n_classes=8, n=None, prefix="train", shuffle=True, normalize=True,
                 outfile="../../inputs/cityscapes/saved_npz/"):
        # C:\\Users\\vinograd\\Documents\\GitHub\\Seg-Grad-CAM\\inputs\\cityscapes\\saved_npz\\
        #self.path = path
        self.scale = scale
        self.n_classes = n_classes
        self.n = n
        self.prefix = prefix
        self.shuffle = shuffle
        self.normalize = normalize
        outfile = Path(outfile)
        outfile.mkdir(parents=True, exist_ok=True)
        if self.normalize:
            self.outfile = os.path.join(outfile,
                                    "scale{}_cl{}_n{}_shuffle{}_{}.npz".format(self.scale, self.n_classes, self.n,
                                                                        self.shuffle, self.prefix))
        else:
            self.outfile = os.path.join(outfile,
                                    "scale{}_cl{}_n{}_shuffle{}_{}_nonorm.npz".format(self.scale, self.n_classes, self.n,
                                                                        self.shuffle, self.prefix))

        self.X = []
        self.Y = []

    def label_to_classes(self, y):
        y = y.copy()
        for lab in LABELS:
            y[y == lab.id] = lab.categoryId
        return to_categorical(y, num_classes=self.n_classes).astype(np.bool)

    def preproc(self, x):
        if self.normalize:
            x = (x / 255.).astype(np.float32)
        return x

    def load_from_path(self,path):
        root = Path(path) / self.prefix
        print("Root path: ", root)
        fx = np.array(tuple(sorted(root.glob("*/*leftImg8bit.png"))))
        #print("Path to images: ", root.glob("*leftImg8bit.png"))
        print("Length of the set:", len(fx), ". ", str(self.n), " will be loaded.", flush=True)

        if self.shuffle:
            np.random.shuffle(fx)

        fx = fx[:self.n]
        fy = tuple(
            str(f).replace("leftImg8bit.png", "gtFine_labelIds.png").replace("leftImg8bit", "gtFine") for f in fx)
        self.X = np.stack(tuple(
            map(lambda f: self.preproc(zoom(imread(f), (1. / self.scale, 1. / self.scale, 1), order=1)), tqdm(fx))))
        self.Y = np.stack(
            tuple(map(lambda f: self.label_to_classes(zoom(imread(f), (1. / self.scale, 1. / self.scale), order=0,
                                                           prefilter=False)), tqdm(fy))))

        # return self.X, self.Y

    def get_and_save_npz(self,path):
        self.load_from_path(path)
        np.savez(self.outfile, X=self.X, Y=self.Y)

    def load_npz(self):
        # TODO: check if the file exists
        npz = np.load(self.outfile)
        self.X, self.Y = npz['X'], npz['Y']


    #############

def get_cityscapes(path, n=None, scale=2, prefix="train", shuffle=True, n_classes=8):
    """Function: Load train/test/val set of Cityscapes"""

    def label_to_classes(y):
        y = y.copy()
        for lab in LABELS:
            y[y == lab.id] = lab.categoryId
        return to_categorical(y, num_classes=n_classes).astype(np.bool)

    def preproc(x):
        x = (x / 255.).astype(np.float32)
        return x

    root = Path(path) / prefix
    print("Root path: ", root)
    fx = np.array(tuple(sorted(root.glob("*/*leftImg8bit.png"))))
    print("Path to images: ", root.glob("*leftImg8bit.png"))
    print("Length of the set:", len(fx))

    if shuffle:
        np.random.shuffle(fx)

    fx = fx[:n]
    fy = tuple(str(f).replace("leftImg8bit.png", "gtFine_labelIds.png").replace("leftImg8bit", "gtFine") for f in fx)
    X = np.stack(tuple(map(lambda f: preproc(zoom(imread(f), (1. / scale, 1. / scale, 1), order=1)), tqdm(fx))))
    Y = np.stack(tuple(map(lambda f: label_to_classes(zoom(imread(f), (1. / scale, 1. / scale), order=0,
                                                           prefilter=False)), tqdm(fy))))

    return X, Y
