# Source code for project: Parametric image harmonization (CVPR 2023 submission: Semi-supervised Parametric Real-world Image Harmonization)
Source code (Training and testing) for project: Parametric image harmonization (Summer intern project 2022). 

<img src='github_images/Figure_teaser.png'>


The demo was developed by [Ke Wang](people.eecs.berkeley.edu/~kewang) based on [PyGame](https://www.pygame.org/news). Ke was a research scientist intern working with [Michaël Gharbi](http://mgharbi.com/), [He Zhang](https://sites.google.com/site/hezhangsprinter/), [Zhihao Xia](https://likesum.github.io/), and [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/). Our demo was developed for MacBook (local) and can run interavtively on CPUs.

Please contact Ke (kewang@adobe.com/kewang@berkeley.edu) or Michaël (mgharbi@adobe.com) for the manuscript submitted to CVPR 2023. Also, feel free to contact us if you have any question. Our code repo is internally hosted at [here](https://git.azr.adobeitc.com/mgharbi/PIH).

## Setup

- Clone this repo:
```bash
git clone https://git.azr.adobeitc.com/adobe-research/parametric_image_harmonization_demo.git
```

- Download the pretrained model from [here](https://adobe-my.sharepoint.com/:u:/p/kewang/EWx38imIw2NCqYHsWqlRjoYBjyQueSfCpnWsMphBqUuqng?e=vAgnb0) and put it in 

```
parametric_image_harmonization_demo/PIH_ResNet/model
```
- Install dependencies

We create a `installation.sh` to install the dependencies, you need to have [Conda](https://docs.conda.io/en/latest/) installed. Run

```
bash installation.sh
```

## Interative demo

Direct into the folder `cd PIH_demo` and run the following command.

```
python demo_masking_new.py
```

Once the GUI is promoted, select the background, foreground and the pre-trained model. Hit the `Harmonize the image` button to obtain the results.

You can use arrow keys to move the foreground objects.
