# LAC_ORIGINAL

In this folder, you will find three versions of the original LAC code of [Minghoa](https://github.com/hithmh/Actor-critic-with-stability-guarantee):

-   [LAC_ORIGINAL](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_ORIGINAL/LAC_ORIGINAL): The original LAC code as supplied by @panweihit.
-   [LAC_ORIGINAL_SEEDED](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED): :seedling: In this version environment and script seeding has been added.
-   [LAC_ORIGINAL_CLEANED_SEEDED](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_ORIGINAL/LAC_ORIGINAL_CLEANED_SEEDED): :seedling::wastebasket: Similar to the [LAC_ORIGINAL_SEEDED](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/tree/master/LAC_ORIGINAL/LAC_ORIGINAL_SEEDED) version but now the code has been cleaned up a bit (Redundant and Unused) parts have been removed.

## Why do we have these different versions

These seeded versions give us a way to compare the translated code to the original LAC code in a (more) deterministic way (for more information about seeding see [this issue](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/issues/30)). While doing this comparison one, however, has to keep in mind that we can not make the versions fully deterministic.The tables below show what components are deterministic between the different Tensorflow/PyTorch versions.

### Environment output

The output of the environment can be successfully seeded in all versions by using the `ENV_SEED` parameter
in the [variant.py](https://github.com/rickstaa/LAC_TF2_TORCH_TRANSLATION/blob/master/legacy/LAC_ORIGINAL/LAC_ORIGINAL/variant.py) file.

| Version | TF1.13             | TF1.15             | TF2.x              | Pytorch            |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ |
| TF1.13  | -                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TF1.15  | :heavy_check_mark: | -                  | :heavy_check_mark: | :heavy_check_mark: |
| TF2.x   | :heavy_check_mark: | :heavy_check_mark: | -                  | :heavy_check_mark: |
| Pytorch | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | -                  |

### LAC algorithm output

The LAC algorithm contains multiple sources of randomness which can be seeded:

-   The Tensorflow/Numpy and Pytorch random number generators.
-   The network weights initializers.
-   The Gaussian Policy noise.

The tables below show for each of these sources whether two versions have deterministic behaviour
between different Tensorflow/Pytorch versions

#### TF/Numpy/Pytorch random number generators

| Version | TF1.13             | TF1.15             | TF2.x              | Pytorch |
| ------- | ------------------ | ------------------ | ------------------ | ------- |
| TF1.13  | -                  | :heavy_check_mark: | :heavy_check_mark: | :x:     |
| TF1.15  | :heavy_check_mark: | -                  | :heavy_check_mark: | :x:     |
| TF2.x   | :heavy_check_mark: | :heavy_check_mark: | -                  | :x:     |
| Pytorch | :x:                | :x:                | :x:                | -       |

#### Network weights initializers

| Version | TF1.13             | TF1.15             | TF2.x | Pytorch |
| ------- | ------------------ | ------------------ | ----- | ------- |
| TF1.13  | -                  | :heavy_check_mark: | :x:   | :x:     |
| TF1.15  | :heavy_check_mark: | -                  | :x:   | :x:     |
| TF2.x   | :x:                | :x:                | -     | :x:     |
| Pytorch | :x:                | :x:                | :x:   | -       |

The difference between TF1.x and Tf2.x is caused by a change in the [GlorotUniform](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform) initializer.

#### Gaussian Policy Noise

| Version | TF1.13             | TF1.15             | TF2.x | Pytorch |
| ------- | ------------------ | ------------------ | ----- | ------- |
| TF1.13  | -                  | :heavy_check_mark: | :x:   | :x:     |
| TF1.15  | :heavy_check_mark: | -                  | :x:   | :x:     |
| TF2.x   | :x:                | :x:                | -     | :x:     |
| Pytorch | :x:                | :x:                | :x:   | -       |
