# Source code for project: Parametric image harmonization (CVPR 2023 submission: Semi-supervised Parametric Real-world Image Harmonization)
Source code (Training and testing) for project: Parametric image harmonization (Summer intern project 2022). 

<img src='github_images/Figure_teaser.png'>


The code was developed by [Ke Wang](people.eecs.berkeley.edu/~kewang). Ke was a research scientist intern working with [Michaël Gharbi](http://mgharbi.com/), [He Zhang](https://sites.google.com/site/hezhangsprinter/), [Zhihao Xia](https://likesum.github.io/), and [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/) at Adobe research during the summer of 2022.

Please contact Ke (kewang@adobe.com or kewang@berkeley.edu) or Michaël (mgharbi@adobe.com) for the manuscript submitted to CVPR 2023. Also, feel free to contact us if you have any question. We also provide an interactive demo repo, internally hosted at [here](https://git.azr.adobeitc.com/adobe-research/parametric_image_harmonization_demo).


## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- [Conda](https://docs.conda.io/en/latest/) installed


## Setup

- Clone this repo:
```bash
git clone https://git.azr.adobeitc.com/mgharbi/PIH
```

- Download the pretrained model from [here](https://adobe-my.sharepoint.com/:u:/p/kewang/EWx38imIw2NCqYHsWqlRjoYBjyQueSfCpnWsMphBqUuqng?e=vAgnb0) and put it in the folder

```
PIH/pretrained/
```
- Install dependencies

We create a `installation.sh` to install the dependencies, you need to have [Conda](https://docs.conda.io/en/latest/) installed. Run

```
bash installation.sh
```
(essentially install [PyTorch](https://pytorch.org/))

## Dataset

We use a subset of internal dataset (The Cooper Dataset) to train the model (contact [He Zhang](https://sites.google.com/site/hezhangsprinter/) for more details), we host the processed training dataset (with post-inpainting background) on AWS s3 (internal)

```
s3://kewang-adobe74k/LR_data.zip ------- Download it by command: aws s3 cp s3://kewang-adobe74k/LR_data.zip <local dir>
```

We host all of our testing benchmarks/images on AWS s3 (internal)

```
s3://kewang-adobe74k/File_testing_all.zip
```

Downloading the dataset (LR_data.zip) can take around an hour depends on the network speed.

## Training

Our approach uses a two-stream semi-supervised training.

<img src='github_images/Figure_3.png'>

See the script `train_example.sh` for training examples.

Training notes:
- modify `--dir_data` to the path of your unzipped `LR_data/train/`.
- modify `--dir_log` to the directory you want to save the checkpoints.
- argument `--masking` corresponds to shading map module.

Simply run:
```
bash train_example.sh
```
to start the training.

## Testing with trained-model

Once we have a trained model, we can run inference on our testig benchmarks (`File_testing_all.zip`)

`File_testing_all.zip` contains RealHM benchmark, adobe internal benchmark, iharm benchmark

See the script `inference_scripts/Inference_Composite_masking_3.sh` for testing examples.

Testing notes:

- modify `--datadir` to the path of `File_testing_all/` folder 

Run:
```
bash inference_scripts/Inference_Composite_masking_3.sh <trained model dir> <target dir> <gpu id> 0.5 <benchmark name>
```

example:

```
bash inference_scripts/Inference_Composite_masking_3.sh pretrained/ckpt_g39.pth results/ 0 0.5 realhm
```

Or try it out our [demo](https://git.azr.adobeitc.com/adobe-research/parametric_image_harmonization_demo)!

