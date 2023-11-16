# UATD Configuration Files and Python Codes
This repository contains mmdetection configurations and codes for UATD Dataset.

Clone this folder within mmdetection/config and run.


## How to Configure Environment

### Installing Anaconda
First install anaconda from https://www.anaconda.com/download/

To update the latest version of anaconda
```bash
conda update -n base -c defaults conda
```

### Setting-up Conda Environment for MMDetection
Create a seperate conda environment for mmdetection with python 3.8 version
```bash
conda create --name mmdetection python=3.8 -y
```

Install pytorch directly from pytorch channel
```bash
conda install pytorch torchvision -c pytorch
```

