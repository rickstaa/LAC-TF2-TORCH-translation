# LAC_TF2_TORCH_Translation

I used this repository to translate the LAC code of [Minghoa](https://github.com/hithmh/Actor-critic-with-stability-guarantee) into tf2 and Pytorch code. It currently contains the following translations:

-   [LAC_ORIGINAL](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_ORIGINAL): The original LAC code of [Minghoa](https://github.com/hithmh/Actor-critic-with-stability-guarantee) as received from @panweihit.
-   [LAC_TF2](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TF2): The LAC code translated to TF2.
-   [LAC_TF2_GRAPH](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TF2_GRAPH): The LAC code translated to TF2 but now with EAGER mode disabled (Deprecated due to performance issues). This version also works with TF1.15.
-   [LAC_TORCH](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TORCH): The LAC code translated into Pytorch code.

All these solutions will give the same results but will differ in training time. The [LAC_TF2](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TF2) version is currently the fastest solution.

## Usage instructions

Below you will find the general instructions on how to use this package. Additionally, every translation also contains its own README.md.

### Setup the python environment

### Conda environment

From the general python package sanity perspective, it is a good idea to use Conda environments to make sure packages from different projects do not interfere with each other.

To create a Conda env with python3, one runs:

```bash
conda create -n <ENV_NAME> python=<PYTHON_VERSION>
```

To activate the env:

```bash
conda activate <ENV_NAME>
```

### Install dependencies

After you created and activated the Conda environment, you have to install the python dependencies. This can be done using the
following command:

```bash
pip install -r requirements.txt
```

## Train instructions

### Change training parameters

You can change the training parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file. The essential variables are explained below:

-   **ENV_NAME**: The environment in which you want to train your agent.
-   **episodes**: The number of episodes you want the agent to perform.
-   **num_of_policies**: The number of (distinct) agents you want to train.
-   **use_lyapunov**: Whether you want to use the LAC (`use_lyapunov=True`) or SAC (`use_lyapunov=False`) algorithm.
-   **ENV_SEED**: The random seed used for the environment. Set to None if you don't want the environment to be deterministic.
-   **RANDOM_SEED**: The random seed of the rest of the script. Set to None if you do not want the script to be deterministic.
-   **continue_training**: Whether we want to continue training an already trained model.
-   **continue_model_folder**: The path of the model for which you want to continue the training
-   **save_checkpoints**: Store intermediate models.
-   **checkpoint_save_freq**: Intermediate model save frequency.

### Start the training

After you set the right hyperparameter in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file, you can train an
algorithm in a specific folder using the following command:

```bash
python <LAC_VERSION_NAME>/train.py
```

## Inference instructions

### Change training parameters

You can change the inference parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file. The essential variables are explained below:

-   **eval_list**: The names of the agents you want to run the inference for.
-   **which_policy_for_inference**: Which policies of a trained agent you want to use for the inference. Each trained agent can contain multiple policies (see: `num_of_policies` parameter).
-   **num_of_paths_for_eval**: How many paths you want to use during the inference for each policy.

### Start the inference

After you trained an agent, you can evaluate the performance of the algorithm by running
the following command:

```bash
python <LAC_VERSION_NAME>/inference_eval.py --model-name=<MODEL_NAME> --env-name=Ex3_EKF_gyro
```

Alternatively, you can set the `eval_list` and `ENV_NAME` parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file and
run the inference with the shorter command:

```bash
python <LAC_VERSION_NAME>/inference_eval.py
```

## Add new environments

New environments should be added in the [<LAC_VERSION_NAME>/envs](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TF2_GRAPH/envs) folder. After you added a new environment
to this folder, you have to add it to the available environments in the [utils.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/utils.py) file. The currently
available environments are found in the [get_env_from_name](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/66f97571f5273508967c3b19102fa127927147f1/LAC_TF2/utils.py#L12) function. If you did this successfully, you could
train the LAC/SAC agent in your environment by setting it as the `ENV_NAME` in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file.
