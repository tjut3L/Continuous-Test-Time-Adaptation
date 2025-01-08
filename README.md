<<<<<<< HEAD
# Continuous-Test-Time-Adaptation
=======
<<<<<<< HEAD
# Continuous-Test-Time-Adaptation
=======
# Continuous test-time adaptation based on mean teacher networks

## Prerequisites

To use the repository, we provide a conda environment.

```bash
conda update conda
conda env create -f environment.yml
conda activate ctta 
```

Our benchmark, including the **evaluation protocol** and **results** is located [here](classification/benchmark.md). More information can be found in the [paper](https://arxiv.org/abs/2306.00650).

<details open>
<summary>Features</summary>

This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**

  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  - `imagenet_r` [ImageNet-R](https://github.com/hendrycks/imagenet-r)

- **Models**

  - For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) or [timm](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13) can be used.
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.
  - Further models include [ResNet-26 GN](https://github.com/zhangmarvin/memo).

- **Methods**

  - The repository currently supports the following methods: source, norm_test, [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
    [CoTTA](https://arxiv.org/abs/2203.13591), [EATA](https://arxiv.org/abs/2204.02610), [SAR](https://arxiv.org/pdf/2302.12400.pdf).

- **Settings**
  
  - `continual` Train the model on a sequence of domains without knowing when a domain shift occurs.
  - `gradual` Train the model on a sequence of gradually increasing/decreasing domain shifts without knowing when a domain shift occurs.
  - `mixed_domains` Train the model on one long test sequence where consecutive test samples are likely to originate from different domains.

</details>

### Get Started

To run one of the following benchmarks, the corresponding datasets need to be downloaded.

- *CIFAR10-to-CIFAR10-C*: the data is automatically downloaded.
- *CIFAR100-to-CIFAR100-C*: the data is automatically downloaded.
- *ImageNet-to-ImageNet-C*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
- *ImageNet-to-ImageNet-R*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary (see function `complete_data_dir_path`). In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them.

### Run Experiments

We provide config files for all experiments and methods. Simply run the following Python file with the corresponding config file.

```bash
python test_time.py --cfg cfgs/[cifar10_c/cifar100_c/imagenet_c/imagenet_others]/[source/norm_test/tent/eata/cotta/sar].yaml
```

For imagenet_others, the argument CORRUPTION.DATASET has to be passed:

```bash
python test_time.py --cfg cfgs/imagenet_others/[source/norm_test/tent/eata/cotta/sar].yaml CORRUPTION.DATASET [imagenet_r]
```

### Changing Configurations

Changing the evaluation configuration is extremely easy. For example, to run TENT on ImageNet-to-ImageNet-C in the `continual` setting with a ResNet-50 and the `IMAGENET1K_V1` initialization, the arguments below have to be passed. Further models and initializations can be found [here (torchvision)](https://pytorch.org/vision/0.14/models.html) or [here (timm)](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13).

```bash
python test_time.py --cfg cfgs/imagenet_c/tent.yaml MODEL.ARCH resnet50 MODEL.WEIGHTS IMAGENET1K_V1 SETTING continual
```

### Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ TENT [official](https://github.com/DequanWang/tent)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ AR-TTA [official](https://github.com/dmn-sjk/ar-tta)
+ SAR [official](https://github.com/mr-eggplant/SAR)

>>>>>>> e97e5ba (first commit)
>>>>>>> fda5cb7 (first commit)
