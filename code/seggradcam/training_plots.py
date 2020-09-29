import numpy as np
import matplotlib.pyplot as plt
from .metrics import metric_name_str
import os

 # Make one class for plots to avoid long parameter lists?
"""class TrainingPlot:
    def __init__(self):"""

FIGSIZE = (32, 16) # (8,8) for textured mnist
FIGSIZEm = (16, 8) # for metrics and loss
# TODO how to adjust figsize and font sizes?
# save few predictions to dry
def plot_predict_and_gt(model, Ximage_gen, Yimage_gen, index_list, trainingdir, n_classes):

    cm = plt.get_cmap('Spectral', n_classes)

    for ind in index_list:
        plt.figure(figsize=FIGSIZE)
        plt.title('Original image')
        if model.input_shape == (None, None, None, 1):
            #Ximage_orig = Ximage_gen[ind][..., 0]
            plt.imshow(Ximage_gen[ind][..., 0], cmap='gray')
        else:
            #Ximage_orig = Ximage_gen[ind]
            plt.imshow(Ximage_gen[ind])

        #plt.imshow(Ximage_orig, cmap='gray')
        plt.savefig(os.path.join(trainingdir,'orig_image_' + str(ind) + '.png'))

        plt.figure(figsize=FIGSIZE)
        plt.title('Predicted mask')
        pred = model.predict(np.expand_dims(Ximage_gen[ind], 0))
        pred_int = np.argmax(pred, axis=-1)[0, ...]
        plt.imshow(pred_int, vmin=0, vmax=n_classes-1, cmap=cm)
        plt.colorbar(fraction=0.046, pad=0.04, cmap=cm)
        plt.savefig(os.path.join(trainingdir,'predict_image_' + str(ind) + '.png'))

        plt.figure(figsize=FIGSIZE)
        plt.title('Ground truth mask')
        plt.imshow(np.argmax(Yimage_gen[ind], axis=-1), vmin=0, vmax=n_classes, cmap=cm)
        plt.colorbar(fraction=0.046, pad=0.04, cmap=cm)
        plt.savefig(os.path.join(trainingdir,'gt_image_' + str(ind) + '.png'))


# plot loss and other metrics

def plot_loss(fit_out, trainingdir):

    plt.figure(figsize=FIGSIZEm)
    plt.title("Loss curve")
    plt.plot(fit_out.history["loss"], label="loss")
    plt.plot(fit_out.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(fit_out.history["val_loss"]), np.min(fit_out.history["val_loss"]), marker="x", color="r",
             label="lowest loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(trainingdir + '/loss.png')


def plot_metric(metrics_list, fit_out, trainingdir):

    for metric in metrics_list:
        met_str = metric_name_str(metric)
        val_str = "val_" + met_str

        plt.figure(figsize=FIGSIZEm)
        plt.title("Learning curve of %s" %met_str)
        plt.plot(fit_out.history[met_str], label=met_str)
        plt.plot(fit_out.history[val_str], label="validation "+met_str)
        plt.plot(np.argmax(fit_out.history[val_str]), np.max(fit_out.history[val_str]), marker="x",
                 color="r", label="highest "+ met_str)
        plt.xlabel("Epochs")
        plt.ylabel(met_str)
        plt.legend()
        plt.savefig(trainingdir + '/' + met_str + '.png')

