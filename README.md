# CHA-CFL

This repository is part of the source code for paper(currently early access version):

​ [Class Hierarchy Aware Contrastive Feature Learning for Multi-Granularity SAR Target Recognition](https://ieeexplore.ieee.org/document/10582307).

The code is conducted under official PyTorch library, and pre-trained models are available for CHA-CFL. Hoping it's helpful!

## Update

_Due to the requirements of PLA, currently we only open-source the pre-trained models. We will open source the overall model as soon as it's available._

- 2024-07-07: update the testing code.
- 2024-07-09: upload the pre-trained models.

## Introduction

![alt text](figures/framework.png)

To fully explore the label distribution of images, we use CHA-CFL to find the hierarchical similarity in categories. The source code of Multi-view Feature Extractor is included in this repository.

## Result

The experiments are carried under MSTAR dataset and shows inspiring results. The result under SOC is shown below.

|    Method    |    10%     |    20%     |    30%     |    40%     |    50%     |
| :----------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|  A-ConvNet   |   76.89%   |   88.31%   |   94.00%   |   95.82%   |   95.85%   |
| RotANet-m2R1 |   83.94%   | **93.05%** |   96.04%   |   97.09%   |   97.95%   |
|  ConFeDent   |   81.36%   |   90.68%   |   94.23%   |   96.95%   |   97.16%   |
|   CHA-CFL    | **84.21%** |   92.66%   | **97.73%** | **98.40%** | **98.85%** |

Notice that the accuracy in the paper is the average accuracy of three random conducted experiments. The provided model test results may slightly deviate from the accuracy results reported in the paper.

## Get Started

The key environment we use are as below:

- GPU: 2070 Super
- CUDA: 12.1
- PyTorch: 1.13.0
- Numpy: 1.26.3

First download the MSTAR dataset from this [link](https://github.com/Owen302266/CHA-CFL/releases/download/model/soc.zip), unzip the package and put the soc folder under dataset folder as follows.It's also need to put pretrained models under the pretrained_models folder.

```
    CHA-CFL
    |   .gitignore
    |   LICENSE
    |   README.md
    |
    +---dataset
    |   |
    |   \---soc
    |       +---test
    |       \---raw
    |
    +---figures
    |       framework.png
    |
    +---pretrained_models
    |       SamplingRate_50_dataset_soc_view_3.pth
    |
    \---src
        |   test.py
        |   validation.ipynb
        |
        +---data
        |   |   loader.py
        |   |   preprocess.py
        |   \---__init__.py
        |
        +---models
        |   |   network.py
        |   |   _base.py
        |   |   _blocks.py
        |   \---__init__.py
        |
        \---utils
            |   common.py
            \---__init__.py

```

We provide both python file and jupyter notebook file for testing. When running the code, only sampling_rate variable is needed to change within[10, 20, 30, 40, 50].

## Pretrained Models

| dataset | dataset sampling rate(%) | accuracy(%) |                                                   link                                                    |
| :-----: | :----------------------: | :---------: | :-------------------------------------------------------------------------------------------------------: |
|   soc   |            10            |    84.58    |                                                 as below                                                  |
|   soc   |            20            |    92.45    |                                                 as below                                                  |
|   soc   |            30            |    97.69    |                                                 as below                                                  |
|   soc   |            40            |    98.43    |                                                 as below                                                  |
|   soc   |            50            |    98.80    | [pre-trained_models](https://github.com/Owen302266/CHA-CFL/releases/download/model/pretrained_models.zip) |

## Contact

If you have any questions, please feel free to contact the authors. Both the e-mails below are available.

- [wenzaidao@nwpu.edu.cn](mailto:wenzaidao@nwpu.edu.cn)
- [wangzikai@mail.nwpu.edu.cn](mailto:wangzikai@mail.nwpu.edu.cn)(recommended)

## Acknowledgement

- Our code structure is build under [AConvNet-pytorch](https://github.com/jangsoopark/AConvNet-pytorch) folder system. It's also available to download the datasets from this [link](https://github.com/jangsoopark/AConvNet-pytorch/releases/download/v2.2.0/dataset.zip).
- The inspiration of Class Hierarchy Aware Contrastive Learning is from [hierarchicalContrastiveLearning](https://github.com/salesforce/hierarchicalContrastiveLearning). We rewrite the loss and applied in our structure.

## Citation

If you find our work is useful in your research, please consider citing:

```tex
@ARTICLE{10582307,
  author={Wen, Zaidao and Wang, Zikai and Zhang, Jianting and Lv, Yafei and Wu, Qian},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  title={Class Hierarchy Aware Contrastive Feature Learning for Multi-Granularity SAR Target Recognition},
  year={2024},
  volume={},
  number={},
  pages={1-17},
  keywords={Feature extraction;Synthetic aperture radar;Training;Task analysis;Data mining;Aerospace and electronic systems;Vectors;Contrastive feature learning;hierarchical classification;multi-granularity target recognition;synthetic aperture radar (SAR)},
  doi={10.1109/TAES.2024.3421171}}

```
