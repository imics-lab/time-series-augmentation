
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
> The main code lies in the src folder. There are 4 different datasets each with a separate sub-directory under src/Medical_Data.<br />
> > Except the Human Activity Recognition data, the other datasets are stored in local machine and will be shared upon request.<br />
> In each of the folders, there is code for lstm, cnn and tst which contains the code for training with LSTM_FCN, InceptionTime and TST for all the traditional datasets.<br />
> The Vae code works to train a Variational AutoEncoder. The signal folder contains the code for the diffusion model.<br />
> There is a separate TTS GAN folder which contains the code for training and testing GAN on all the datasets.  <br />
>  
>
> 
> 



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

