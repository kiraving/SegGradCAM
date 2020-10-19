############
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import numpy as np
import keras
#from keras.optimizers import Adam
from .metrics import *


class TrainingParameters:
    """The class stores necessary parameters for training a model,
    writing a description file about an experiment, and further visualizations."""
    def __init__(self,
                 dataset_name = 'Cityscapes',
                 n_classes=8
                ,scale = 2
                ,batch_size = 2
                ,last_activation = 'softmax'
                ,n_depth = 4
                ,n_filter_base = 32  # 16
                ,pool = 2
                ,lr = 3.e-4
                ,epochs = 100
                ,validation_steps = 200
                ,steps_per_epoch = 600
                ,loss = "categorical_crossentropy"
                ,optimizer = "keras.optimizers.Adam" #(lr=3.e-4)" #Adam(lr=3.e-4) #eval(f"keras.optimizers.Adam(lr={lr})")
                ,metrics = ['accuracy']#,iou_coef,dice_coef] # names for iou and dice as str?
                ,n_train = 2975  # 2975 max for cityscapes, alternatively set to None
                ,n_val = 500  # 500 max for cityscapes, alternatively set to None
                ,outfolder = ''
                 , input_shape=(None, None, 3)
                ):
        self.n_classes = n_classes
        self.scale = scale
        self.batch_size = batch_size
        self.last_activation = last_activation
        self.n_depth = n_depth
        self.n_filter_base = n_filter_base
        self.pool = pool
        self.lr = lr
        self.epochs = epochs
        self.validation_steps = validation_steps
        self.steps_per_epoch = steps_per_epoch
        self.loss = loss
        self.optimizer = eval(f"{optimizer}(lr={lr})")
        self.metrics = [custom_metric(metric) for metric in metrics]
        self.n_train = n_train  # 2975 max
        self.n_val = n_val  # 500 max
        self.dataset_name = dataset_name
        self.input_shape = input_shape

        self.timestamp = datetime.now()
        timestr = self.timestamp.strftime("%m_%d_%H_%M")

        if outfolder == '':
            outfolder = Path("../../output") / f"{dataset_name}"/f"{timestr}_fil{n_filter_base}_depth{n_depth}_lr{lr}_scale{scale}_batch{batch_size}"
            outfolder.mkdir(exist_ok=True, parents=True)
        self.outfolder = str(outfolder)

        self.param_dict = dict(n_classes = n_classes
        ,scale = scale
        ,batch_size = batch_size
        ,last_activation = last_activation
        ,n_depth = n_depth
        ,n_filter_base = n_filter_base
        ,pool = pool
        ,lr = lr
        ,epochs = epochs
        ,validation_steps = validation_steps
        ,steps_per_epoch = steps_per_epoch
        ,loss = loss
        ,optimizer = f"{optimizer}(lr={lr})"
        ,metrics = metrics
        ,n_train = n_train
        ,n_val = n_val
        ,dataset_name = dataset_name
        ,input_shape = input_shape
        ,outfolder = str(outfolder)
        )

    def saveToJson(self): #TODO how to update existing json?
        with open(os.path.join(Path(self.outfolder),'Parameters.json'), 'w') as json_file:
            json.dump(self.param_dict, json_file)

    def loadFromJson(self):
        with open(os.path.join(self.outfolder, 'Parameters.json'), 'r') as json_file:
            self.param_dict=json.load(json_file)
        self.scale = self.param_dict['scale']
        self.n_classes = self.param_dict['n_classes']
        self.outfolder = self.param_dict['outfolder']
        self.batch_size = self.param_dict['batch_size']
        self.last_activation = self.param_dict['last_activation']
        self.n_depth = self.param_dict['n_depth']
        self.n_filter_base = self.param_dict['n_filter_base']
        self.pool = self.param_dict['pool']
        self.lr = self.param_dict['lr']
        self.epochs = self.param_dict['epochs']
        self.validation_steps = self.param_dict['validation_steps']
        self.steps_per_epoch = self.param_dict['steps_per_epoch']
        self.loss = self.param_dict['loss']
        self.optimizer = eval(self.param_dict['optimizer'])
        self.metrics = [custom_metric(metric) for metric in self.param_dict['metrics']]
        self.n_train = self.param_dict['n_train']
        self.n_val = self.param_dict['n_val']
        self.dataset_name = self.param_dict['dataset_name']
        self.input_shape = self.param_dict['input_shape']

    def __str__(self):
        return "{}".format(self.param_dict)

        #TODO how to update attributes according to the loaded values

    #def saveToMetaResults(self):
    # add a line about results into cityscapes_train_results.csv

class TrainingResults: #(TrainingParameters,tf.keras.callbacks.History):

    def __init__(self, trainingParameters, fit_out):
        self.end_time = datetime.now()
        self.params = trainingParameters
        self.fit_out = fit_out

    def modelSummaryTxt(self):
        with open(Path(self.params.outfolder) / 'modelsummary.txt', 'w+') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.fit_out.model.summary(positions=[.33, .6, .80, 1.], print_fn=lambda x: fh.write(x + '\n'))
            fh.close()

    def writeResultsTxt(self):
        with open(Path(self.params.outfolder) / 'TrainingResults.txt', 'w+') as fh:
            timedif = (self.end_time - self.params.timestamp) / timedelta(minutes=1)
            fh.write("\nTraining time (min): " + str(timedif))
            fh.write("\nlowest val_loss " + str(np.min(self.fit_out.history["val_loss"])))
            for metric in self.params.metrics:
                met_str = metric_name_str(metric)
                val_str = "val_" + met_str
                fh.write("\nbest validation  " + met_str + " "+str(np.max(self.fit_out.history[val_str])))
            fh.close()

        #TODO what else to write to the file? what info is necessary for a paper?

        #TODO write info about data augmentation


