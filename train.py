import torch
import tqdm
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from model import SimpleCNN, count_parameters, save_model
from torch.optim.lr_scheduler import StepLR

def show_transformed_images(data_loader):
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Display a few images
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(6):
        ax = axes[i]
        img = images[i].squeeze(0)  # Remove the channel dimension for grayscale images
        img = img.numpy()
        ax.imshow(img, cmap="gray")
        ax.axis('off')
    plt.show()


def train_model():

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
      # Train Phase transformations
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomRotation((-10, 10), fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std are tuples
    ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

       # Show some transformed images
    print("Transformed Training Images:")
    show_transformed_images(train_loader)
    model =  SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.046, momentum=0.90,weight_decay=0.001)
    scheduler=StepLR(optimizer,step_size=4,gamma=0.1)
    #optimizer=optim.Adam(model.parameters(), lr=0.055, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

    EPOCHS = 10
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []   
    for epoch in range(EPOCHS):
     print("Epoch:",epoch)
     train(model, device, train_loader, optimizer, epoch)
     test(model, device, test_loader)
     scheduler.step()
    
    # Initialize model


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  cri=torch.nn.CrossEntropyLoss()
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
    #print(target)
    # Predict
    #print(target.size())
    y_pred = model(data)
    # Calculate loss
    
    #loss = F.nll_loss(y_pred, target)
    loss=cri(y_pred,target)
    # train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    print(f'Train Accuracy: {accuracy:.2f}%')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    map=[]
    test_loss = 0
    correct = 0
    cri=torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss+=cri(output,target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    
    test_loss /= len(test_loader.dataset)
    #test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    # Save model with timestamp
    model_filename = save_model(model, accuracy)
    model_filename=''
    
    return model, accuracy, model_filename

if __name__ == '__main__':
    train_model()
