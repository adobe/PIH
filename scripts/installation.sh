conda create -n pytorch_pih python=3.9
conda activate pytorch_pih
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install matplotlib
pip install opencv-python
pip install tqdm
