import models
import torch
import utils
import os

in_tensor = torch.randn((2,3,224,224))

model1 = models.ResNet50(10)
model2 = models.ResNet101(10)
model3 = models.ResNet152(10)

y1 = model1(in_tensor)
y2 = model2(in_tensor)
y3 = model3(in_tensor)

if y1.shape == (2, 10) and y2.shape == (2,10) and y3.shape == (2,10):
	print("Model module works")
else:
	print("Model module doesn't work")


dataset_csv = os.path.join('dataset', 'train.csv')
my_dataset = utils.get_dataset(dataset_csv)
my_dataloader = utils.get_dataLoader(my_dataset, 16)

dataiter = iter(my_dataloader)
images, labels = dataiter.next()

if images.shape == (16, 3, 224, 224) and images.dtype == torch.float32:
	print("Dataset and Dataloader work")
else:
	print("Dataset and Dataloader do not work")

for idx, (inputs, labels) in enumerate(my_dataloader):
	if idx == 0:
		print(inputs.shape, labels.shape)
		break
