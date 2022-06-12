from utils import check_valid_loop

def train(
    model, data_loader, 
    criterion, optimizer,
    epoch, total_epochs, 
    device='cuda'
):
    print('TRAINING')
    model.train()
    total_loss = 0.
    for i, (images, target) in enumerate(data_loader):
        images, target = images.to(device), target.to(device)
        
        pred = model(images)
        loss = criterion(pred, target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_loss = total_loss / (i+1)
        if (i+1) % 50 == 0:
            print(
                f"Epoch: {epoch+1}/{total_epochs}, ",
                f"Iteration: {i+1}/{len(data_loader)},",
                f"Loss: {loss.item():.4f}, Average Loss: {average_loss:.4f}"
            )
    return total_loss/len(data_loader)

def validate(model, data_loader, criterion, device='cuda', epoch=None):
    print('VALIDATING')
    validation_loss = 0.0
    model.eval()
    for i,(images, target) in enumerate(data_loader):
        images, target = images.to(device), target.to(device)
        
        pred = model(images)
        check_valid_loop(pred, images, epoch, i)
        loss = criterion(pred,target)
        validation_loss += loss.item()
        if (i+1) % 50 == 0:
            print(
                f"Iteration: {i+1}/{len(data_loader)}", 
                f"Loss: {loss.item():.4f}"
            )
    final_valid_loss = validation_loss / len(data_loader)
    return final_valid_loss