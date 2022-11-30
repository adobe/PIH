conda create -n pytorch_env python=3.9
conda activate pytorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install matplotlib
pip install opencv-python
pip install vit_pytorch
pip install tqdm
