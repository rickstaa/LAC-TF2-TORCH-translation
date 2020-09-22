# TF_REWRITE

I used this repository to translate the LAC code of [Minghoa](https://github.com/hithmh/Actor-critic-with-stability-guarantee) into tf2 and pytorch code. I left it here as a
quick example on how to translate tf1 code to tf2 and pytorch code.

## Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_old python=3.6
```

To activate the env:

```
conda activate lac_old
```

# Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

# filter_LAC_tf_rewrite
