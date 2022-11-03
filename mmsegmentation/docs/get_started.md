## Prerequisites

- Linux
- Python 3.9
- PyTorch == 1.9.0
- CUDA 10.2
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The version of MMSegmentation and MMCV used in this repo are listed as below. Please install the correct version of MMCV to avoid installation issues.

| MMSegmentation version |    MMCV version     |
|:-------------------:|:-------------------:|
| 0.17.0              | [mmcv-full==1.3.9](https://github.com/open-mmlab/mmcv/releases/tag/v1.3.9) |

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
Here we use PyTorch 1.9.0 and CUDA 10.2.
You may also switch to other version by specifying the version number.

```shell
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```

c. Install mmcv-full, please refer to [MMCV](https://mmcv.readthedocs.io/en/latest/) for more details.

```shell
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
```

d. Install MMSegmentation.

```shell
# Assuming that you are in the parent folder of mmsegmentation
cd mmsegmentation
pip install -e .  # or "python setup.py develop"
```

Note:

1. When MMsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
2. Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
   To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

### A from-scratch setup script

#### Linux

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab

conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  # or "python setup.py develop"
```

#### Developing with multiple MMSegmentation versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMSegmentation in the current directory.

To use the default MMSegmentation installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMSegmentation and the required environment are installed correctly, we can run sample python codes to initialize a segmentor and inference a demo image when a coarse mask is given:

```python
cd mmsegmentation/demo
sh run.sh
```

The above code is supposed to run successfully upon you finish the installation.


