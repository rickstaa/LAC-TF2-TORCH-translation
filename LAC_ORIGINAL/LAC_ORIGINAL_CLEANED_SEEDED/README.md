# LAC_TF1_ORIGINAL_CLEANED

The original LAC code as received from @panweihit but now cleaned up. Meaning the
components we currently don't use were removed. This version works with `tf<=1.15`.

## Use instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_original_cleaned_seeded python=3.6
```

To activate the env:

```bash
conda activate lac_original_cleaned_seeded
```

### Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

### Train instructions

After you set the right hyperparameter in the `variant.py` file you can train an
algorithm using the following command:

```bash
python train.py
```

### Evaluation instructions

To evaluate the performance of the algorithm you can use the following command:

```bash
python inference_eval.py --model-name=LAC20201001_1121 --env-name=Ex3_EKF_gyro
```
