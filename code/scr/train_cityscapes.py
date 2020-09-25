import numpy as np

from pathlib import Path
from tqdm import tqdm
from imageio import imread
"""from csbdeep.internals.nets import common_unet
from csbdeep.internals.blocks import unet_block
"""
from csbdeep.utils.tf import CARETensorBoard

from keras.models import Model
from keras.layers import Conv2D, Input, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from scipy.ndimage import zoom

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
scr_folder = os.path.abspath('.')
if scr_folder not in sys.path:
    sys.path.insert(1, scr_folder)

from dataloaders import Cityscapes
from metrics import iou_coef, dice_coef, dice_loss
from unet import csbd_unet, manual_unet, TrainUnet
from training_write import TrainingParameters, TrainingResults
from training_plots import plot_predict_and_gt, plot_loss, plot_metric


path = "C:\\Users\\vinograd\\Documents\\GitHub\\Seg-Grad-CAM\\inputs\\cityscapes\\leftImg8bit_trainvaltest\\leftImg8bit"

trainparam = TrainingParameters(
epochs = 2,
n_train= 10, #2975 max
n_val = 5 # 500 max #"""
,steps_per_epoch = 10 #1400
)
trainparam.saveToJson()

trainset = Cityscapes(path, n = trainparam.n_train, shuffle = True, scale = trainparam.scale, prefix = "train")
valset = Cityscapes(path, n = trainparam.n_val, shuffle = False, scale = trainparam.scale, prefix = "val")
trainset.save_npz()
#trainset.load_npz()
valset.save_npz()

trainunet = TrainUnet(trainparam)
trainunet.csbdUnet()
fit_out = trainunet.fit_generator(trainset, valset)

# save few predictions to dry
str_folder = str(trainparam.outfolder)
plot_predict_and_gt(trainunet.model, valset.X, valset.Y, range(3), str_folder, trainparam.n_classes)
# plot loss and other metrics
plot_loss(fit_out,  str_folder)
plot_metric(trainparam.metrics, fit_out, str_folder)

trainingResults = TrainingResults(trainparam, fit_out)
trainingResults.modelSummaryTxt()
trainingResults.writeResultsTxt()

# add a line about results into cityscapes_train_results.csv

