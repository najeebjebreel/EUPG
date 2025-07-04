import torch
import copy


def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, device = 'cuda', 
                verbose_epoch = 10, patience = None):

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(max_epochs/5), gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    model.to(device)
    model.train()

    best_val_loss = 100000
    consecutive_no_improvement = 0
    best_model = None

    for epoch in range(max_epochs):
        running_loss = 0.0
        for batch in train_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss/= len(train_loader)
        # scheduler.step()

        if epoch == 0 or (epoch + 1) % verbose_epoch == 0:
            val_loss = eval_loss(model, val_loader, criterion, device)
            # print(f'Epoch [{epoch+1}/{max_epochs}] | Train Loss: {running_loss} | Validation Loss: {val_loss}')    
        
        # Check for early stopping
        if patience is not None: 
            if val_loss >= best_val_loss:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= patience:
                    # print(f'Validation loss has not improved for {patience} consecutive epochs. Stopping early.')
                    return best_model
            else:
                consecutive_no_improvement = 0
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
    
    return model


# Validation loss
def eval_loss(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                features, labels = val_batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        return val_loss/len(val_loader)



def train_tabnet_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, device = 'cuda', 
                verbose_epoch = 10, patience = None, idxs = None):

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(max_epochs/5), gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    model.to(device)
    model.train()

    best_val_loss = 100000
    consecutive_no_improvement = 0
    best_model = None

    for epoch in range(max_epochs):
        running_loss = 0.0
        for batch in train_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            if idxs is not None:
                outputs = model(features[:, idxs[0]], features[:, idxs[1]].long())
            else:
                outputs = model(features)
            loss = criterion(outputs, labels)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        running_loss/= len(train_loader)
        # scheduler.step()

        if epoch == 0 or (epoch + 1) % verbose_epoch == 0:
            val_loss = eval_loss(model, val_loader, criterion, device, idxs=idxs)
            # print(f'Epoch [{epoch+1}/{max_epochs}] | Train Loss: {running_loss} | Validation Loss: {val_loss}')    
        
        # Check for early stopping
        if patience is not None: 
            if val_loss >= best_val_loss:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= patience:
                    # print(f'Validation loss has not improved for {patience} consecutive epochs. Stopping early.')
                    return best_model
            else:
                consecutive_no_improvement = 0
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
    
    return model


# Validation loss
def eval_tabnet_loss(model, val_loader, criterion, device, idxs=None):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                features, labels = val_batch
                features, labels = features.to(device), labels.to(device)
                if idxs is not None:
                    outputs = model(features[:, idxs[0]], features[:, idxs[1]].long())
                else:
                    outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        return val_loss/len(val_loader)


