import torch
import torch.optim as optim
import torch.nn as nn
from model import Net
import dataset


learning_rate = 0.01
epochs = 2

net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(dataset.trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:
      print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
      running_loss = 0.0

print('Training Finished')

torch.save(net.state_dict(), 'net.pth')

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

  print(f'Accuracy on the 10000 test images: {100 * correct / total}')

test()
