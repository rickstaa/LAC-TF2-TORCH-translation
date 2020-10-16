# LAC_TF2_GRAPH

In this folder the original LAC code has been translated to be compatible with TF2 and TF1.15.
This was done using the steps given in the
[tensorflow migration guide](https://www.tensorflow.org/guide/migrate). This version
however does not yet work in eager mode and therefore uses the
`tf.compat.v1.disable_eager_execution()` flag.

## Usage instructions

Below you will find the instructions on how to use this package.

### Setup the python environment

### Conda environment

From the general python package sanity perspective, it is a good idea to use Conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs

```bash
conda create -n lac_tf2 python=3.8
```

To activate the env:

```bash
conda activate lac_tf2
```

### Install dependencies

After you created and activated the Conda environment, you have to install the python dependencies. This can be done using the following command:

```bash
pip install -r requirements.txt
```

If you want to use TF1.15 instead of TF2 you can use the following command:

```bash
pip install -r requirements_tf115.txt
```

## Train instructions

### Change training parameters

You can change the training parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/variant.py) file. The essential variables are explained below:

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

After you set the right hyperparameter in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/variant.py) file, you can train an
algorithm in a specific folder using the following command:

```bash
python LAC_TF2_GRAPH/train.py
```

## Inference instructions

### Change training parameters

You can change the inference parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/variant.py) file. The essential variables are explained below:

-   **eval_list**: The names of the agents you want to run the inference for.
-   **which_policy_for_inference**: Which policies of a trained agent you want to use for the inference. Each trained agent can contain multiple policies (see: `num_of_policies` parameter).
-   **num_of_paths_for_eval**: How many paths you want to use during the inference for each policy.

### Start the inference

After you trained an agent, you can evaluate the performance of the algorithm by running
the following command:

```bash
python LAC_TF2_GRAPH/inference_eval.py --model-name=<MODEL_NAME> --env-name=Ex3_EKF_gyro
```

Alternatively, you can set the `eval_list` and `ENV_NAME` parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/variant.py) file and
run the inference with the shorter command:

```bash
python LAC_TF2_GRAPH/inference_eval.py
```

## Add new environments

New environments should be added in the [LAC_TF2_GRAPH/envs](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_TF2_GRAPH/envs) folder. After you added a new environment
to this folder, you have to add it to the available environments in the [utils.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/utils.py) file. The currently
available environments are found in the [get_env_from_name](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/d02e543fa35132645e81b69aa2fc6c1f07af1dff/LAC_TF2_GRAPH/utils.py#L12) function. If you did this successfully, you could
train the LAC/SAC agent in your environment by setting it as the `ENV_NAME` in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2_GRAPH/variant.py) file.
