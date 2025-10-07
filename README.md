### System Setup (for WSL)
Run these once before installing Python dependencies:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit cudnn-cuda-12
nvcc --version
nvidia-smi
sudo apt-get update
sudo apt-get install -y cudnn-cuda-12
dpkg -l | grep cudnn
```
Then create and activate your conda environment:
```bash
conda create -n GAR-project 
conda activate GAR-project
conda install pip
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

### (Optional) Setup you command line interface for better readability
```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

