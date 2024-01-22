from tqdm import tqdm
import torch.nn as nn
import torch

# Train model with given hyperparameters and data
def train_model(model, optimizer, num_epochs, train_dataloader, val_dataloader, device):
    """Train the model
    Parameters
    ----------
    model : torch.nn.Module
        model to train.
    optimizer : torch.optim
        optimizer.
    num_epochs : int
        number of epochs.
    train_dataloader : torch.utils.data.DataLoader
        dataloader for training.
    val_dataloader : torch.utils.data.DataLoader
        dataloader for validation.
    device : torch.device
        device to use.
    
    Returns
    ------- 
    history : dict
        dictionary containing the loss history for training and validation.
    """
    history = {'train' : [],
               'validation' : []}
    
    model = model.to(device)

    for epoch in range(num_epochs):

        running_loss = 0.0
        mean_loss = 0.0

        model.train()
        
        # Train step
        with tqdm(train_dataloader, unit="batches") as tepoch:
            
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                loss = nn.functional.mse_loss(pred, target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                mean_loss = running_loss / (batch_idx + 1)
                
                tepoch.set_postfix(loss="{:.6f}".format(mean_loss))
        
        history['train'].append(mean_loss)

        # Validation step
        if val_dataloader != None:
            val_running_loss = 0.0
            val_mean_loss = 0.0
            model.eval()

            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(val_dataloader):
                    val_data = val_data.to(device)
                    val_target = val_target.to(device)
                    pred = model(val_data)
                    val_loss = nn.functional.mse_loss(pred, val_target)
                    val_running_loss += val_loss.item()
                    val_mean_loss = val_running_loss / (batch_idx + 1)

            print("{:.{}f}".format(val_mean_loss, 6))

            history['validation'].append(val_mean_loss)

    return history



        