
<div align="center">


<!-- Title-->
# Time Series Augmentation: In-depth analysis with Biomedical Data

<!-- BADGES -->
> <div align="left">


</div>


<!-- DESCRIPTION -->
## Description
>
>  
> This repository servers as a codebase for various augmentations that can be applied to time series data. In this repository, we have provided code for traditional as well as deep learning basd augmentations. Testing and results are generated with four different datasets.
> 
> 


<!-- SETUP -->
## Setup

Before running the code, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch
- Nvidia CUDA toolkit and cuDNN (for GPU acceleration)

```bash
pip install torch torchvision
conda install cudatoolkit
```


### Conda Virtual Environment

Create the Conda virtual environment using the [environment file](environment.yml):
```bash
conda env create -f environment.yml

# dynamically set python path for the environment
conda activate time_series_augmentation
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


<!-- USAGE -->
## Usage
> `### <<< DELETE ME:` ***Usage***
>  
> Provide information on how to run your project. Delete the below example.
> 
> `### DELETE ME >>>`

```python
from foo import bar

bar.baz("hello world")
```

```bash
python -m foo.bar "hello world"
```


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation
```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  year={Year}
}
```

