import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as s 
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Convolutional Neural Network
class Net3a(nn.Module):
	def __init__(self):
		super(Net3a, self).__init__()
		self.w = nn.Linear(32*32*3, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = self.w(x)
		return x

class Net3b(nn.Module):
	def __init__(self):
		super(Net3b, self).__init__()
		self.w = nn.Linear(32*32*3, 800)
		self.r = nn.Linear(800, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = F.relu(self.w(x))
		x = self.r(x)	
		return x

M = 800 #higher
p = 6 #chill
N = 5 #lower
class Net3c(nn.Module):
	def __init__(self):
		super(Net3c, self).__init__()
		self.w = nn.Linear((((33-p) // N)**2) * M, 10)
		self.conv = nn.Conv2d(3, M, p)
		self.max_pool = nn.MaxPool2d(N)

	def forward(self, x):
		x = self.conv(x)
		x = F.relu(x)	
		x = self.max_pool(x)
		x = x.view(-1, (((33-p) // N)**2) * M)
		x = self.w(x)
		return x

#net = Net3a()
#net = Net3b()
net = Net3c()

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

#Load data to test the network
dataiter = iter(testloader)
images, labels = dataiter.next()

e = 12
train_acc = np.zeros(e)
test_acc = np.zeros(e)
index = 0
index2 = 0
#Train the network
for epoch in range(e):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

	#How does network perform on the entire train dataset
	correct = 0
	total = 0
	with torch.no_grad():
			for data in trainloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
	train_acc[index] = 100.0 * correct / total
	print(100.0 * correct / total)
	index = index + 1
	#How does network perform on the entire dataset
	correct = 0
	total = 0
	with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
	test_acc[index2] = 100.0 * correct / total
	index2 = index2 + 1
print('Finished Training')

print("train accuracy: ")
print(train_acc)
print("test accuracy: ")
print(test_acc)
iterations = [1,2,3,4,5,6,7,8,9,10,11,12]
s.set()
train_line, = plt.plot 
plt.plot(iterations, train_acc)
plt.plot(iterations, test_acc)
plt.xlabel("Iterations")
plt.xticks((1,2,3,4,5,6,7,8,9,10, 11, 12))
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.show()


