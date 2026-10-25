import torch
from loader import trainloader
from optimizer import optimizer, criterion
from net import net

PATH = './cifar_net.pt'

if __name__ == '__main__':
   device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
   net.to(device)

   for epoch in range(2):
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
         # get the inputs; data is a list of [inputs, labels]
         inputs, labels = data[0].to(device), data[1].to(device)

         # zero the parameter gradients
         optimizer.zero_grad()

         # forward + backward + optimize
         outputs = net(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         # print statistics
         running_loss += loss.item()
         if i % 2000 == 1999: # print every 2000 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

   print('Finished Training')

   torch.save(net.state_dict(), PATH)
