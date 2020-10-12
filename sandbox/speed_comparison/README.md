# TF2_TORCH_SAMPLE_SPEED_COMPARISON

Small test repository to analysis if there is a speed difference between pytorch and
tensorflow when sampling from the normal distribution.

## Usage instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n tf_torch_speed_test python=3.8
```

To activate the env:

```bash
conda activate tf_torch_speed_test
```

### Installation Environment

```bash
pip install -r requirements.txt
```
