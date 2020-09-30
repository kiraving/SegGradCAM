from keras import backend as K
"""
def my_metric_factory(the_param=1.0):
    def fn(y_true, y_pred):
        return my_dummy_metric(y_true, y_pred, the_param=the_param)

    fn.__name__ = 'metricname_{}'.format(the_param)
    return fn
"""
from keras.utils.generic_utils import serialize_keras_object
# Jaccard coefficient = IoU
def IoU(smooth=1):
    def iou_coef(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
    iou_coef.__name__ = 'IoU'
    def __str__(self):
        return 'IoU'
    return iou_coef
#serialize_keras_object(IoU())

def Dice(smooth=1):
    def dice_coef(y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice
    dice_coef.__name__ = 'Dice'
    def __str__(self):
        return 'Dice'
    return dice_coef

# Dice can be a loss as well.
def dice_loss(y_true, y_pred):
    return 1-Dice( smooth=1)


def custom_metric(name):
    if 'iou' in name.lower(): #str(name).lower():
        return IoU(smooth=1)
    if 'dice' in name.lower():
        return Dice(smooth=1)
    else:
        return name


def metric_name_str(metric):
    """retrieves a name of metric for plots"""
    if metric == 'accuracy':
        met_str = 'acc'
    elif 'iou' in str(metric).lower():
        met_str = 'IoU'
    elif 'dice' in str(metric).lower():
        met_str = 'Dice'
    else:
        print("Unknown metric")  # TODO learn how to raise errors
    return met_str