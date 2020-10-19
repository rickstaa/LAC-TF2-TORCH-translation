# LAC_TF1_ORIGINAL

The original LAC code as received from @panweihit. In this version I added a optional random_seed to the weight initialization and policy noise.Nothing was changed in this version. This version works with `tf<=1.15`.

## Usage instructions

### Setup the python environment

### Conda environment

From the general python package sanity perspective, it is a good idea to use Conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_original_seeded python=3.6
```

To activate the env:

```bash
conda activate lac_original_seeded
```

### Installation Environment

After you created and activated the Conda environment, you have to install the python dependencies. This can be done using the following command:

```bash
pip install -r requirements.txt
```

## Train instructions

### Change training parameters

You can change the training parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file. The essential variables are explained below:

-   **env_name**: The environment in which you want to train your agent.
-   **episodes**: The number of episodes you want the agent to perform.
-   **num_of_policies**: The number of (distinct) agents you want to train.
-   **ENV_SEED**: The random seed used for the environment. Set to None if you don't want the environment to be deterministic.
-   **RANDOM_SEED**: The random seed of the rest of the script. Set to None if you do not want the script to be deterministic.

### Start the training

After you set the right hyperparameter in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file, you can train an
algorithm in a specific folder using the following command:

```bash
python ./legacy/LAC_ORIGINAL/LAC_ORIGINAL/train.py
```

## Inference instructions

### Change training parameters

You can change the inference parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file. The essential variables are explained below:

-   **eval_list**: The names of the agents you want to run the inference for.
-   **which_policy_for_inference**: Which policies of a trained agent you want to use for the inference. Each trained agent can contain multiple policies (see: `num_of_policies` parameter).
-   **num_of_paths_for_eval**: How many paths you want to use during the inference for each policy.

### Start the inference

After you trained an agent, you can evaluate the performance of the algorithm by running
the following command:

```bash
python ./legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/inference_eval.py --model-name=<MODEL_NAME> --env-name=Ex3_EKF_gyro
```

Alternatively, you can set the `eval_list` and `ENV_NAME` parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file and
run the inference with the shorter command:

```bash
python LAC_TF2_GRAPH/inference_eval.py
```

## Add new environments

New environments should be added in the [LAC_TF2_GRAPH/envs](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/envs) folder. After you added a new environment
to this folder, you have to add it to the available environments in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file. The currently
available environments are found in the [get_env_from_name](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/96ca8f311ebc30e6b25a7af66b102cb37c599e14/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py#L193) function. If you did this successfully, you could
train the LAC/SAC agent in your environment by setting it as the `ENV_NAME` in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED/variant.py) file.
