import numpy as np
from pathlib import Path
from .seggradcam import SegGradCAM,BiasRoI, SuperRoI, ClassRoI, PixelRoI
from matplotlib import pyplot as plt
import os, sys
from operator import sub
import pandas as pd
from .visualize_sgc import SegGradCAMplot
from .training_plots import plot_predict_and_gt

def propToEachActivation(cls, roi, roi_type, activations=None, image=None, trainunet=None, next_dict=None, image_id=None,
                         gt=None, n_classes=None, outfolder=None, model=None):
    """create a folder for the experiment. Save there original image, ground truth, predicted mask,
    seg-grad-cam's explanations for each activation.
    Either trainunet or n_classes, outfolde & model should be provided.
    Image & gt or next_dict & image_id should be given.
    If Seg-Grad-CAM is needed for only one component (e.g. ClassRoi.largestComponent is called),
    then set roi_type to 'Roi'

    How to run it on TexturedMnist:
    image_id = 2
    cls = 2
    propToEachActivation(cls, roi=ClassRoI(trainunet.model, next_dict[0][image_id], cls),
                         roi_type='Class',trainunet=trainunet, next_dict=next_dict, image_id=image_id)"""

    if trainunet != None:
        n_classes = trainunet.trainparam.n_classes
        outfolder = str(trainunet.trainparam.outfolder)
        model = trainunet.model
        activations = trainunet.activationLayers()
    # if nones raise error
    from datetime import datetime

    timestamp = datetime.now()
    timestr = timestamp.strftime("%m%d%H%M")
    outfolder = Path(os.path.join(outfolder, f"all_act_cls{cls}_{roi_type}_{timestr}"))
    outfolder.mkdir(exist_ok=True, parents=True)
    # create new folder for an experiment

    # save gt, predict, original
    if next_dict != None and image_id != None:
        plot_predict_and_gt(model, next_dict[0], next_dict[1], [image_id], outfolder, n_classes)
        image = next_dict[0][image_id]

    for act in activations[:-1]:
        pixsgc = SegGradCAM(model, image, cls, act, activations[-1], roi=roi,  # PixelRoI(i,j,image),
                            normalize=True, abs_w=False, posit_w=False)
        pixsgc.SGC()
        plotter = SegGradCAMplot(pixsgc,  # trainunet=trainunet,
                                 next_dict=next_dict, image_id=image_id, gt=gt,
                                 n_classes=n_classes, outfolder=outfolder, model=model)
        if roi_type == 'Pixel':
            plotter.explainPixel()
        elif roi_type == 'Class':
            plotter.explainClass()
        else:
            plotter.explainRoi()


def SlidingPixel():

    pass

def ProportionBias(nextset, model, n_samp, prop_to_layer,roi=SuperRoI()):
    Debug = False
    #n_samp = 5
    propor_sal = np.zeros((n_samp, 10))

    from IntegratedGradients import integrated_gradients
    ig = integrated_gradients(model)  # , outchannels=[2])
    predicted = model.predict(nextset[0])

    for i in range(n_samp):
        a = nextset[2][i]['digit_with_infill'][..., 0]
        b = nextset[2][i]['biased_tile'][..., 0]
        c = sub(b, a)
        # print(c.shape)
        c = np.ones(c.shape) * [c > 0]  # np.max(c,0)
        B = c[0]
        # print(B.nonzero()[1].shape)
        Ad = a.nonzero()[0].shape[0]
        Ad = Ad / c.shape[2] / c.shape[1]  # proportion of area under the digit
        Ab = B.nonzero()[0].shape[0] / c.shape[2] / c.shape[1]
        # plt.imshow(B)
        # plt.colorbar()
        c = sub(1 - b, a)
        # print(c.shape)
        c = np.ones(c.shape) * [c > 0]  # np.max(c,0)
        U = c[0]
        # print(U.nonzero()[0].shape,c.shape)
        Au = U.nonzero()[0].shape[0] / c.shape[2] / c.shape[1]
        image, predictions = nextset[0][i], np.argmax(predicted[i], axis=2)
        #prop_to_layer = 'activation_' + str(int(model.layers[-1].name.split('_')[1]) - lcount // 2)  # 9
        #prop_from_layer = model.layers[-1].name
        # gt= np.argmax(ground_truth[im], axis=2)
        cls = i % 10
        # gradcamdir1img=save_dir+'/'+datetime.now().strftime("%d-%m-%H-%M")+'randomtest'+str(im)+'/'
        caminst= SegGradCAM(model, image, cls, prop_to_layer, prop_from_layer='last', roi)
        cam = caminst.SGC()
        Sb = cam * B
        nansumcam = np.nansum(cam)
        print("SGC for class: ", cls)
        print("Biased bgr: ", np.nansum(Sb), np.max(Sb), np.nansum(Sb) / Ab)
        Su = cam * U
        print("Unbiased bgr: ", np.nansum(Su), np.max(Su), np.nansum(Su) / Au)
        Sd = cam * a
        print("Digit: ", np.nansum(Sd), np.max(Sd), np.nansum(Sd) / Ad)

        igex = ig.explain(image, outc=cls, reference=np.ones(image.shape) * 0.5  # image.mean()
                          , num_steps=200, verbose=1)[..., 0]
        absmax = np.max(igex)
        Sbig = igex * B
        print("IG for class: ", cls)
        print("IG Biased bgr: ", np.nansum(Sbig), np.max(Sbig), np.max(Sbig) / absmax, np.nansum(Sbig) / Ab)
        Suig = igex * U
        print("IG Unbiased bgr: ", np.nansum(Suig), np.max(Suig), np.max(Sbig) / absmax, np.nansum(Suig) / Au)
        Sdig = igex * a
        print("IG Digit: ", np.nansum(Sdig), np.max(Sdig), np.max(Sbig) / absmax, np.nansum(Sdig) / Ad)
        # absnansum = np.absolute(np.nansum(Sbig)/Ab)+np.absolute(np.nansum(Suig)/Au)+np.absolute(np.nansum(Sdig)/Ad)
        absnansum = np.nansum(igex)  # np.absolute(igex))
        print("absnansum for IG: ", absnansum)

        print("absmax for IG: ", absmax)
        new_rescaler = np.nansum(Sb / Ab) + np.nansum(Su / Au) + np.nansum(Sd / Ad)
        propor_sal[i] = [cls, np.nansum(Sb) / Ab / new_rescaler, np.nansum(Su) / Au / new_rescaler,
                         np.nansum(Sd) / Ad / new_rescaler, np.nansum(Sb) / nansumcam, np.nansum(Su) / nansumcam,
                         np.nansum(Sd) / nansumcam, np.nansum(Sbig) / absnansum, np.nansum(Suig) / absnansum,
                         np.nansum(Sdig) / absnansum]
        print("area proportions: ", Ab, Au, Ad, Ab + Au + Ad)
        # normalize to 100%
        # what to do with negatives?

    df = pd.DataFrame(propor_sal, columns=["cls", "SalB/AreaB", "SalU/AreaU", "SalD/AreaD", "SalBias", "SalUnbias",
                                           "SalDigit", "InGradBias", "InGradU",
                                           "InGradD"])  # rescaled, then unscaled
    df = df.fillna(0)

    return propor_sal, df

# TODO: max, average, sum & etc of b-neck layers -> explain layer choice