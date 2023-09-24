# SEG-GRAD-CAM
Publicly available implementation in Keras of our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/7244) "Towards Interpretable Semantic Segmentation via Gradient-Weighted Class Activation Mapping" by Kira Vinogradova, Alexandr Dibrov, Gene Myers.

Check out our [poster](./poster_Vinogradova_AAAI_Feb2020.pdf) for a schematic overview of the method.

No updates or upgrades to newer versions are planned.  

# Installation
``pip install git+https://github.com/kiraving/SegGradCAM.git``

# Requirements
Python 3.6, (recommended) Anaconda, versions of other packages can be found [here](./code/get_versions.ipynb)

Please download [Cityscapes](https://www.cityscapes-dataset.com/) (Fine annotations) if you intend to test Seg-Grad-CAM on a real-world dataset collected on German roads.

# Usage
* [Code for Seg-Grad-CAM method](./seggradcam/seggradcam.py#L118)
* [Notebook for training, loading pretrained model and usage of Seg-Grad-CAM on TextureMNIST](./code/textureMNIST-notebooks/demo.ipynb)
* [Training a U-Net with a backbone on Cityscapes & applying Seg-Grad-CAM](./code/cityscapes-notebooks/city_demo_backbone.ipynb) 
* [Vanilla U-Net on Cityscapes & Seg-Grad-CAM](./code/cityscapes-notebooks/city_demo_vanilla.ipynb)

# Credits:
[CSBDeep](https://github.com/csbdeep/csbdeep)

    @article{weigert2018content,
      title={Content-aware image restoration: pushing the limits of fluorescence microscopy},
      author={Weigert, Martin and Schmidt, Uwe and Boothe, Tobias and M{\"u}ller, Andreas and Dibrov, Alexandr and Jain, Akanksha and Wilhelm, Benjamin and Schmidt, Deborah and Broaddus, Coleman and Culley, Si{\^a}n and others},
      journal={Nature methods},
      volume={15},
      number={12},
      pages={1090--1097},
      year={2018},
      publisher={Nature Publishing Group}
    }

[Cityscapes](https://www.cityscapes-dataset.com/) dataset:

    @inproceedings{Cordts2016Cityscapes,
    title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
    author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
    booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016}
    }

[segmentation_models](https://github.com/qubvel/segmentation_models) package:

     @misc{Yakubovskiy:2019,
        Author = {Pavel Yakubovskiy},
        Title = {Segmentation Models},
        Year = {2019},
        Publisher = {GitHub},
        Journal = {GitHub repository},
        Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
      }

[TextureMNIST dataset](https://github.com/boschresearch/GridSaliency-ToyDatasetGen)
Code for toy dataset generation of "Grid Saliency for Context Explanations of Semantic Segmentation" [paper](https://arxiv.org/abs/1907.13054)

# How to cite Seg-Grad-CAM:

    @inproceedings{Vinogradova2020TowardsIS,
      title={Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping},
      author={Kira Vinogradova and Alexandr Dibrov and Eugene W. Myers},
      booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
      year      = {2020},
      doi       = {10.1609/aaai.v34i10.7244}
    }
