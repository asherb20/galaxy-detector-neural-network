import torch
import torchvision
from loader import testloader, classes, batch_size
from imshow import imshow
from net import net
from train import PATH

def compare_images():
   dataiter = iter(testloader)
   images, labels = next(dataiter)

   # print images
   imshow(torchvision.utils.make_grid(images))
   print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

   net.load_state_dict(torch.load(PATH, weights_only=True))

   outputs = net(images)

   _, predicted = torch.max(outputs, 1)

   print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

if __name__ == '__main__':
   # prepare to count predictions in total
   correct = 0
   total = 0

   # prepare to count predictions for each class
   correct_pred = {classname: 0 for classname in classes}
   total_pred = {classname: 0 for classname in classes}

   # since we're not training, we don't need to calculate the gradients for our outputs
   with torch.no_grad():
      for data in testloader:
         images, labels = data
         # calculate outputs by running images through the network
         outputs = net(images)
         # the class with the highest energy is what we choose as prediction
         _, predictions = torch.max(outputs, 1)

         # collect the total number of predictions and correct predictions
         total += labels.size(0)
         correct += (predictions == labels).sum().item()

         # collect the correct predictions for each class
         for label, prediction in zip(labels, predictions):
            if label == prediction:
               correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

   # print accuracy for whole network
   print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

   # print accuracy for each class
   for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')