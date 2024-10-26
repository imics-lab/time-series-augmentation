
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
### File Organization
> 1.  The main code lies in the src folder. There are 4 different datasets each with a separate sub-directory under src/Medical_Data.<br />
> 2. Except the Human Activity Recognition data, the other datasets are stored in local machine and will be shared upon request.<br />
> 3. In each of the folders, there is code for lstm, cnn and tst which contains the code for training with LSTM_FCN, InceptionTime and TST for all the traditional datasets.<br />
> 4. The Vae code works to train a Variational AutoEncoder. The signal folder contains the code for the diffusion model.<br />
> 5. There is a separate TTS GAN folder which contains the code for training and testing GAN on all the datasets.  <br />
### Running code
> #### To run an experiment with a dataset:
> #### Example dataset: Human Activity Recognition data
> 1. Go to src/Medical_Data/UniMibB_SHAR
> 2. Run jupyter notebook UniMiB_SHAR_ADL_load_datset.ipynb. This will automatically download the dataset into the folder.
> #### TRADITIONAL AUGMENTATION
> 3. Run python files unimib_LSTM_test.py, unimib_CNN_test.py, unimib_TST_test.py
> 4. This will print the results of the best augmentaion hyperparameter for each of the traditional augmentations.
> 5. Open the final_LSTM_unimib.py, final_CNN_unimib.py, final_TST_unimib.py, in the main function type in the hyperarameters generated from the previous step in the respective files.
> 6. The variables to update are jitter_val, scale_val, mag_warp_val, time_warp_val, window_warp_val, window_slice_val.
> 7. Run the python files final_LSTM_unimib.py, final_CNN_unimib.py, final_TST_unimib.py.
> 8. This will give the accuracy and 95% confidence interval results for the traditional augmentation methods.
> #### VARIATIONAL AUTOENCODER
> 9. Run unimib_vae.py to generate numpy arrays of the trained model
> 10. Run unimib_vae_test.py to generate the results of classification using samples generated from trained vae model.
> #### DIFFUSION
> 11. Go to folder /signal
> 12. Copy the dataset folder to this location, or else running the following code will download the data again.
> 13. Run ddpm1d_cls_free.py, this wil train the diffusion model.
> 14. After training, run final_unimib_diffusion.py, this will use the trained model to generate new samples and augment them with the existing dataset. The new files will be saved as numpy arrays.
> 15. Run final_unimib_diffusion_new.py to get the final accuracy results along with the confidence intervals.
> #### GENERATIVE ADVERSARIAL NETWORK
> 16. Go to src/TTS_GAN
> 17. Copy the contents of src/TTS_GAN/unimib to src/unimib
> 18. Run unimib_train_TTS_GAN.py to train the GAN model.
> 19. Run unimib_final_gan_new.py to generate samples with the trained model and then classify them with the three classifiers.
> 20. It will output the 10 different accuracy values for the 10 different shuffles. Calculate the average accuracy and 95% confidence interval using the values
> #### Finished experiment with 1 dataset.
> Similarly follow steps with other three datasets. However, the data is stored locally due to size restrictions. It will be shared upon request.
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
@inproceedings{de2024impact,
  title={The Impact of Data Augmentation on Time Series Classification Models: An In-Depth Study with Biomedical Data},
  author={De, Bikram and Sakevych, Mykhailo and Metsis, Vangelis},
  booktitle={International Conference on Artificial Intelligence in Medicine},
  pages={192--203},
  year={2024},
  organization={Springer}
}
```

