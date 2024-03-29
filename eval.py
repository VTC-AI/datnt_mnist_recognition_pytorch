import torch
import dataset
from model import Net


net = Net()
net.load_state_dict(torch.load('net.pth'))

def test():
  correct = 0
  total = 0
  with torch.no_grad():
    for data in dataset.testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  print(f'Accuracy on 10000 test images: {100 * correct / total}')

  test()
