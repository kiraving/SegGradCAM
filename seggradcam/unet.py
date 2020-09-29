from csbdeep.internals.blocks import unet_block
from keras.models import Model
from keras.layers import Conv2D, Input, Activation, Conv2DTranspose, BatchNormalization, Dropout, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import os
import sys
from pathlib import Path
"""scr_folder = os.path.abspath('.')
if scr_folder not in sys.path:
    sys.path.insert(1, scr_folder)
"""
from .training_write import TrainingParameters, TrainingResults
from .dataloaders import Cityscapes
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from csbdeep.utils.tf import CARETensorBoard

# Unet from CSBDeep

def csbd_unet(input_shape = (None, None, 3), last_activation = "softmax", n_classes=8, n_depth=3, n_filter_base = 32,
                     batch_norm = True, pool = 4):

    inp = Input(input_shape)
    lay = unet_block(n_depth=n_depth,n_filter_base =n_filter_base,
                     batch_norm = batch_norm,
                     pool = (pool,pool),
                     last_activation = "relu")(inp)
    out = Conv2D(n_classes,(1,1), padding = "same")(lay)
    out = Activation(last_activation)(out)

    return Model(inp, out)

###
# Manually constructed Unet

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def manual_unet(input_img, bottleneck_depth=5, n_classes=3, n_filters=16, dropout=0.05,
                      batchnorm=True, last_act='softmax'):

    # bottleneck_dept can be [2,3,4,5]
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if bottleneck_depth > 2:
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        if bottleneck_depth > 3:
            p3 = MaxPooling2D((2, 2))(c3)
            p3 = Dropout(dropout)(p3)

            c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
            if bottleneck_depth > 4:
                p4 = MaxPooling2D(pool_size=(2, 2))(c4)
                p4 = Dropout(dropout)(p4)

                c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

                # expansive path
                u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
                u6 = concatenate([u6, c4])
                u6 = Dropout(dropout)(u6)
                c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

                u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
            else:
                u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4)
            u7 = concatenate([u7, c3])
            u7 = Dropout(dropout)(u7)
            c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
            u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        else:
            u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    else:
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c2)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c10 = Conv2D(n_classes, (1, 1))(c9)
    outputs = Activation(last_act)(c10)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def encoder_decoder(input_img, bottleneck_depth=5, n_classes=3, n_filters=16,
                                 dropout=0.05, batchnorm=False, last_act='softmax'):
    """An encoder-decoder architecture analogical to Unet architecture, but without short-cut connections.
    bottleneck_dept can be 2,3,4,or 5"""

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    if bottleneck_depth > 2:
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        if bottleneck_depth > 3:
            p3 = MaxPooling2D((2, 2))(c3)
            p3 = Dropout(dropout)(p3)

            c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
            if bottleneck_depth > 4:
                p4 = MaxPooling2D(pool_size=(2, 2))(c4)
                p4 = Dropout(dropout)(p4)

                c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

                # expansive path
                u6 = UpSampling2D(size=(2, 2))(c5)
                u6 = Dropout(dropout)(u6)
                c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

                u7 = UpSampling2D(size=(2, 2))(c6)
            else:
                u7 = UpSampling2D(size=(2, 2))(c4)
            u7 = Dropout(dropout)(u7)
            c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

            u8 = UpSampling2D(size=(2, 2))(c7)
        else:
            u8 = UpSampling2D(size=(2, 2))(c3)
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        u9 = UpSampling2D(size=(2, 2))(c8)
    else:
        u9 = UpSampling2D(size=(2, 2))(c2)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    c10 = Conv2D(n_classes, (1, 1))(c9)
    outputs = Activation(last_act)(c10)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


class TrainUnet:
    def __init__(self, trainingParameters):
        self.trainparam = trainingParameters
        self.cb = []
        self.model = None
        self.fit_out = None
        self.datagen = ImageDataGenerator(
            #rescale=1. / 255,
            # brightness_range = (.8,1.2),
            horizontal_flip=True)

    def csbdUnet(self):
        self.model = csbd_unet(input_shape=self.trainparam.input_shape, last_activation=self.trainparam.last_activation
                               , n_classes=self.trainparam.n_classes, n_depth=self.trainparam.n_depth,
                               n_filter_base=self.trainparam.n_filter_base, pool=self.trainparam.pool)
    def manualUnet(self):
        # keras Input?
        """self.model = manual_unet(input_img, bottleneck_depth=5, n_classes=3, n_filters=16, dropout=0.05,
                      batchnorm=True, last_act='softmax')
            csbd_unet(input_shape=(None, None, 3), last_activation=self.trainparam.last_activation
                               , n_classes=self.trainparam.n_classes, n_depth=self.trainparam.n_depth,
                               n_filter_base=self.trainparam.n_filter_base, pool=self.trainparam.pool)
"""
        pass
    def encoderDecoder(self):
        pass

    def compile(self):
        self.model.compile(loss=self.trainparam.loss, optimizer=self.trainparam.optimizer, metrics=self.trainparam.metrics)

    #@staticmethod
    def dataAugment(self):
        """self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            # brightness_range = (.8,1.2),
            horizontal_flip=True)"""
        pass


    def fit_generator(self, trainset, valset):
        #datagen = self.dataAugment()
        dry = self.trainparam.outfolder
        self.cb.append(ModelCheckpoint(f"{dry}/weights.h5", save_best_only=True, save_weights_only=True))
        params = dict(prefix_with_timestamp=False, n_images=3, write_images=True)
        if self.trainparam.dataset_name == 'Cityscapes':
            self.cb.append(CARETensorBoard(f"{dry}", **params,
                                           output_slices=[[slice(None), slice(None), slice(None), slice(1, 8, 3)]]))

        self.compile()
        if self.trainparam.dataset_name == 'Cityscapes':
            train_input = self.datagen.flow(trainset.X, trainset.Y, batch_size=self.trainparam.batch_size)
            # don't use generator for validation data that always produces different images in each epoch
            # make numpy arrays for val data -> needed for CARETensorBoard to show images
            val_input = self.datagen.flow(valset.X, valset.Y, batch_size=self.trainparam.batch_size*self.trainparam.validation_steps)
            val_input = val_input[0]
            validation_steps = None
        else:
            train_input = trainset
            val_input = valset
            validation_steps = self.trainparam.validation_steps

        self.fit_out = self.model.fit_generator(train_input,
                                      validation_data=val_input,
                                      epochs=self.trainparam.epochs,
                                      steps_per_epoch=self.trainparam.steps_per_epoch,
                                      validation_steps=validation_steps,
                                      callbacks=self.cb)
        return self.fit_out

    def load_weights(self):
        self.model.load_weights(Path(self.trainparam.outfolder) / "weights.h5")


    def activationLayers(self):
        activations = []
        for ll in self.model.layers:
            if 'activation' in ll.name:
                activations.append(ll.name)
        return activations

    def findBottleneck(self):
        lcount = 0
        for ll in self.model.layers:
            if 'activation' in ll.name:
                lcount += 1
                #print(lcount, ll.name)
        mid_act = int(self.model.layers[-1].name.split('_')[1]) - lcount // 2
        #print('middle act: activation_' + str(mid_act))
        print('activations in b-neck: ', mid_act - 1, ',', mid_act)

        return ['activation_' + str(mid_act-1), 'activation_' + str(mid_act)]

    def predict1Image(self, image):
        #if self.trainparam.dataset_name == 'Cityscapes':
            #image = image/255.
        pred = trainunet.model.predict(np.expand_dims(image, 0))
        return np.argmax(pred, axis=-1)[0, ...]
