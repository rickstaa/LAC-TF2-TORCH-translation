# Cleaned up version of the LAC script (tf>2.0 - eager mode)

This folder contains the cleaned up version of the LAC script. It uses `pytorch>=1.6.0`.

## Use instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_torch python=3.8
```

To activate the env:

```bash
conda activate lac_torch
```

### Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

### Usage instructions

#### Train instructions

After you set the right hyperparameter in the `variant.py` file you can train an
algorithm using the following command:

```bash
python train.py
```

#### Evaluation instructions

To evaluate the performance of the algorithm you can use the following command:

```bash
python inference_eval.py --model-name=LAC20201001_1121 --env-name=Ex3_EKF_gyro
```

## Use GPU

If you want to use GPU you have to install Pytorch using the following command:

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```
