# Cleaned up version of the LAC script (tf>2.0 - eager mode)

This folder contains the cleaned-up version of the LAC script. It uses `pytorch>=1.6.0`.

## Usage instructions

Below you will find the instructions on how to use this version.

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

## Use GPU

If you want to use GPU you have to install Pytorch using the following command:

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```

You can then enable GPU computing by setting the `USE_GPU` variable in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_Translation/blob/f492ceb1ede9c22e5f4fae45085f2393465aeb61/LAC_TORCH/variant.py#L12) file.

## Train instructions

### Change training parameters

You can change the training parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TORCH/variant.py) file. The essential variables are explained below:

-   **ENV_NAME**: The environment in which you want to train your agent.
-   **EPISODES**: The number of episodes you want the agent to perform.
-   **NUM_OF_POLICIES**: The number of (distinct) agents you want to train.
-   **USE_LYAPUNOV**: Whether you want to use the LAC (`use_lyapunov=True`) or SAC (`use_lyapunov=False`) algorithm.
-   **ENV_SEED**: The random seed used for the environment. Set to None if you don't want the environment to be deterministic.
-   **RANDOM_SEED**: The random seed of the rest of the script. Set to None if you do not want the script to be deterministic.
-   **CONTINUE_TRAINING**: Whether we want to continue training an already trained model.
-   **CONTINUE_MODEL_FOLDER**: The path of the model for which you want to continue the training
-   **SAVE_CHECKPOINTS**: Store intermediate models.
-   **CHECKPOINT_SAVE_FREQ**: Intermediate model save frequency.

### Start the training

After you set the right hyperparameter in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TORCH/variant.py) file, you can train an
algorithm in a specific folder using the following command:

```bash
python <LAC_VERSION_NAME>/train.py
```

## Inference instructions

### Change training parameters

You can change the inference parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TF2/variant.py) file. The essential variables are explained below:

-   **EVAL_LIST**: The names of the agents you want to run the inference for.
-   **WHICH_POLICY_FOR_INFERENCE**: Which policies of a trained agent you want to use for the inference. Each trained agent can contain multiple policies (see: `num_of_policies` parameter).
-   **NUM_OF_PATHS_FOR_EVAL**: How many paths you want to use during the inference.

### Start the inference

After you trained an agent, you can evaluate the performance of the algorithm by running
the following command:

```bash
python LAC_TORCH/inference_eval.py --model-name=<MODEL_NAME> --env-name=Ex3_EKF_gyro
```

Alternatively, you can set the `eval_list` and `ENV_NAME` parameters in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/LAC_TORCH/variant.py) file and
run the inference with the shorter command:

```bash
python LAC_TORCH/inference_eval.py
```

## Add new environments

New environments should be added to the `ENVS_PARAMS` variable of the
[variant.py file](https://github.com/rickstaa/LAC_TF2_TORCH_Translation/blob/f492ceb1ede9c22e5f4fae45085f2393465aeb61/LAC_TORCH/variant.py#L108-L141). While doing so please make sure you supply a valid
module and class name:

```python
"oscillator": {
    "module_name": "envs.oscillator",
    "class_name": "oscillator",
    "max_ep_steps": 800,
    "max_global_steps": TRAIN_PARAMS["episodes"],
    "max_episodes": int(1e6),
    "eval_render": False,
},
```

After that, the new environment can be used by setting the `ENV_NAME` variable of the
of the [variant.py file](https://github.com/rickstaa/LAC_TF2_TORCH_Translation/blob/f492ceb1ede9c22e5f4fae45085f2393465aeb61/LAC_TF2/variant.py#L20).
