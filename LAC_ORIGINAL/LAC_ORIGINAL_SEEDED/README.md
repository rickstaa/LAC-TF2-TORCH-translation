# LAC_TF1_ORIGINAL

The original LAC code as received from @panweihit. Nothing was changed in this version.
This version works with `tf<=1.15`.

## Usage instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac python=3.6
```

To activate the env:

    conda activate lac

### Installation Environment

```bash
pip install numpy==1.16.3
pip install tensorflow==1.13.1
pip install tensorflow-probability==0.6.0
pip install opencv-python
pip install cloudpickle
pip install gym
pip install matplotlib
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

### Train instructions

After you set the right hyperparameter in the `variant.py` file you can train an
algorithm using the following command:

```bash
python main.py
```

### Evaluation instructions

To evaluate the performance of the algorithm you can use the following command:

```bash
python inference_eval.py --model-name=LAC20201001_1121 --env-name=Ex3_EKF_gyro
```
