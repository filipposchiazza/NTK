import torch
import torch.nn as nn
import os
import pickle


class MLP(nn.Module):
    
    def __init__(self, input_dim, num_layers, num_channels):
        """Multi-layer perceptron
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        num_layers : int
            Number of layers.
        num_channels : int
            Number of channels.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_channels = num_channels

        layers = [nn.Linear(input_dim, num_channels), nn.ReLU()]
        for _ in range(1, num_layers - 1):
            layers.extend([nn.Linear(num_channels, num_channels), nn.ReLU()])
        layers.extend([nn.Linear(num_channels, 3), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)



    def forward(self, x):
        "Forward pass"
        return self.network(x)
    

    
    def predict(self, x):
        """Predict the output
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        Returns
        -------
        pred : torch.Tensor
            Output tensor.
        """
        with torch.no_grad():
            pred =self.network(x)
        return pred
    


    def get_num_parameters(self):
        """Get the number of parameters
        
        Returns
        -------
        num_parameters : int
            Number of parameters.
        """
        num_parameters = 0
        for p in self.parameters():
            num_parameters += p.numel()
        return num_parameters


    # Saving methods
    def save_model(self, save_folder):
        """Save the parameters and the model state_dict
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where the model will be saved.
        
        Returns
        -------
            None.
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        param_file = os.path.join(save_folder, 'parameters.pkl')
        parameters = [self.input_dim,
                      self.num_layers,
                      self.num_channels]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
    
        model_file = os.path.join(save_folder, 'model.pt')
        torch.save(self.state_dict(), model_file)



    @staticmethod
    def save_history(history, save_folder):
        """Save the training history
        
        Parameters
        ----------
        history : dict
            Dictionary containing the training history.
        save_folder : str
            Path to the folder where the history will be saved.
        """
        filename = os.path.join(save_folder, 'history.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(history, f)



    # Loading methods
    @classmethod
    def load_model(cls, save_folder, map_location='cpu'):
        """Load the model and its parameters
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where the model is saved.
        map_location : str, optional
            Where to load the model. The default is 'cpu'.
        
        Returns
        -------
        model : MLP
            The loaded model.
        """
        param_file = os.path.join(save_folder, 'parameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
            
        model = cls(*parameters)
        
        model_file = os.path.join(save_folder, 'model.pt')
        model.load_state_dict(torch.load(model_file, map_location=map_location))
        
        return model
    


    @staticmethod
    def load_history(save_folder):
        """Load the training history
        
        Parameters
        ----------
        save_folder : str
            Path to the folder where the history is saved.
        
        Returns
        -------
        history : dict
            Dictionary containing the training history."""
        history_file = os.path.join(save_folder, 'history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history