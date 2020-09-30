# Cleaned up version of the LAC script (tf>2.0 - eager mode)

In this folder the original LAC code has been translated to be compatible with tf2.
This was done using the steps given in the
[tensorflow migration guide](https://www.tensorflow.org/guide/migrate). In this version
eager mode is enabled.

## Use instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_clean_tf2_eager python=3.8
```

To activate the env:

```bash
conda activate lac_clean_tf2_eager
```

### Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.
