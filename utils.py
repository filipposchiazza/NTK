import torch
import matplotlib.pyplot as plt

# Fourier feature mapping
def input_mapping(x, B):
    """Map the input to the Fourier feature space
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    B : torch.Tensor
        Matrix of random parameters.
    
    Returns
    -------
    torch.Tensor
        Mapped tensor."""
    if B is None:
        return x
    else:
        x_proj = torch.matmul(2. * torch.pi * x, B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    


# Get the B matrix for the Fourier features mapping
def get_B_gauss(sigma, mapping_size, seed=0):
    """Get the B matrix for the Fourier features mapping, with Gaussian distribution
    
    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.
    mapping_size : int
        Number of features.
    seed : int, optional
        Seed for the generation of the matrix. The default is 0.
        
    Returns
    -------
    torch.Tensor
        B matrix."""
    # include a seed for the generation
    torch.manual_seed(seed)
    B_gauss = torch.randn(mapping_size, 2) * sigma
    return B_gauss


def save_B_matrix(B, save_folder):
    """Save the B matrix
    
    Parameters
    ----------
    B : torch.Tensor
        B matrix.
    save_folder : str
        Path to the folder where the B matrix will be saved.
    
    Returns
    -------
        None.
    """
    torch.save(B, save_folder + 'B.pt')



def plot_results(original_img, validation_dataset, model, device):
    """Plot the prediction and the original image
    
    Parameters
    ----------
    original_img : np.ndarray
        Original image.
    validation_dataset : torch.utils.data.Dataset
        Validation dataset.
    model : torch.nn.Module
        Trained model.
    device : torch.device
        Device.
        
    Returns
    -------
        None.
    """
    pred = model.predict(validation_dataset[:][0].to(device)).reshape(original_img.shape).cpu()
    # Plot pred e img side by side
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title('Prediction')
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    plt.title('Original')
    plt.show()