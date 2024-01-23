# NTK - Multi-Layer Neural Tangent Kernel
This repository contains the implementation of the Multi-Layer Neural Tangent Kernel NTK (https://arxiv.org/pdf/2006.10739.pdf).

## Installation
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```
Alternatively, you can install the dependencies manually:

```bash
pip install .
```

## Usage
The main components of this project are located in the modules directory:

-  **MLP_model.py**: Contains the MLP class, which is a multi-layer perceptron model. This file also includes functions to save and load the model and its training history.
- **train.py**: Contains the train_model function, which trains a given model with specified hyperparameters and data.
- **utils.py**: Contains utility functions for Fourier feature mapping, saving and loading the B matrix, and plotting results.
- **dataset.py**: Contains the FourierFeaturesPixelDataset, which is a dataset that maps image's pixels to Fourier features.

To train a model, you can use the train_model function from train.py. This function requires a MLP model, an optimizer, the number of epochs, a training dataloader, a validation dataloader, and a device.

```python
from modules.train import train_model
from modules.MLP_model import MLP
import torch.optim as optim

model = MLP(input_dim=2*config.MAPPING_SIZE,
            num_layers=config.NETWORK_NUM_LAYERS,
            num_channels=config.NETWORK_CHANNELS)

optimizer = optim.Adam(model.parameters(),
                       lr=config.LEARNING_RATE)


history = train_model(model, 
                      optimizer, 
                      num_epochs, 
                      train_dataloader, 
                      val_dataloader, 
                      device)
```

You can save and load the model and its training history using the save_history, load_model, and load_history methods of the MLP class in MLP_model.py.

```python
model.save_model(save_folder)
model.save_history(history, save_folder)
# To load the model
model_loaded = MLP.load_model(save_folder)
# To load the history
history_loaded = MLP.load_history(save_folder)
```

The utils.py file contains functions for Fourier feature mapping (input_mapping and get_B_gauss), saving the B matrix (save_B_matrix), and plotting results (plot_results).

```python
import modules.utils as utils

# sigma is the standard deviation of the Gaussian distribution and mapping_size  represents the number of Fourier features
B = get_B_gauss(sigma, mapping_size)
mapped_tensor = utils.map_input_tenso(input_tensor=coords,
                                      B=B)
# To save the B matrix
save_B_matrix(B, save_folder)
# To plot the results
plot_results(original_img, validation_dataset, model, device)
```

The notebook example.ipynb is a tutorial on how to use the model and the functions in this project.
