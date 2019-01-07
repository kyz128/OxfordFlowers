#imports 
import argparse
import torch
from torch import nn, optim 
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import transforms, models, datasets
import json


def data_preprocess(img, resize_factor, crop_factor):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	train_transforms = transforms.Compose([transforms.Resize(resize_factor),
										   transforms.CenterCrop(crop_factor),
										   transforms.RandomHorizontalFlip(),
										   transforms.RandomVerticalFlip(),
										   transforms.ToTensor(),
										   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

	test_transforms = transforms.Compose([transforms.Resize(resize_factor),
	                                      transforms.CenterCrop(crop_factor),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

	train_dataset = datasets.ImageFolder(root=train_dir, transform= train_transforms)
	test_dataset= datasets.ImageFolder(root=valid_dir, transform= test_transforms)

	valid_size = 0.2
	batch_size= 32
	num_test = len(test_dataset)
	indices = list(range(num_test))
	np.random.shuffle(indices)
	split = int(np.floor(valid_size * num_test))
	valid_idx, test_idx = indices[split:], indices[:split]

	test_sampler = SubsetRandomSampler(test_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
	valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=valid_sampler)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
	print('Finished preprocessing dataset')
	return train_loader, valid_loader, test_loader, train_dataset


def choose_model(model_name, hidden_units):
    if model_name== 'vgg16':
        model= models.vgg16(pretrained=True)
        for param in model.features.parameters():
    		param.requires_grad= False
		model.classifier= nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                		nn.ReLU(),
                                		nn.Dropout(0.5),
                                		nn.Linear(hidden_units, hidden_units),
                                		nn.ReLU(),
                                		nn.Dropout(0.5),
                                		nn.Linear(hidden_units, 102),
                                		nn.LogSoftmax(dim=1))
    elif model_name== 'inception':
    	model= models.inception_v3(pretrained=True)
    		for param in model.parameters():
    			param.requires_grad= False    	
		model.AuxLogits.fc = nn.Linear(768, 102)
		model.fc = nn.Linear(2048, 102)
		ct = []
		for name, child in inception.named_children():
    		if "Conv2d_4a_3x3" in ct:
	        	for params in child.parameters():
	            	params.requires_grad = True
    		ct.append(name)

	return model

def save_model(model, name, dataset, optimizer, hidden_units):
	class_to_idx= dataset.class_to_idx
	checkpoint = {'arch': name,
				  'hidden_units': hidden_units,
	              'class_to_idx': class_to_idx,
	              'state_dict': model.state_dict(),
	              'opt_dict': optimizer.state_dict()}
	torch.save(checkpoint, 'checkpoint.pth')
	print('Model saved at checkpoint.pth')

def train(epoch, model, model_name, train_loader, valid_loader, criterion, optimizer, hidden, gpu)
	n_epoch= epoch
	min_loss= np.inf
	train_on_gpu= gpu and torch.cuda.is_available()
	for epoch in range(1, n_epoch+1):
	    train_loss= 0.0
	    valid_loss= 0.0
	    model.train()
	    for img, label in train_loader:
	        if train_on_gpu:
	            img, label = img.cuda(), label.cuda()
	        optimizer.zero_grad()
	        if model_name== 'vgg16':
	        	output= model(img)
	        	loss= criterion(output, label)
	        elif model_name== 'inception':
	        	outputs, aux_outputs = inception(img)
        		loss1 = criterion(outputs, label)
        		loss2 = criterion(aux_outputs, label)
        		loss = loss1 + 0.4*loss2
	        loss.backward()
	        optimizer.step()
	        train_loss+= loss.item() * img.size(0)

	    model.eval()
    	correct=0
    	with torch.no_grad()
		for img, label in valid_loader:
		  if train_on_gpu:
		      img, label = img.cuda(), label.cuda()
		  output= model(img)
		  loss= criterion(output, label)
		  valid_loss+= loss.item() * img.size(0)
		  prob = torch.exp(output)
		  top_prob, top_class = prob.topk(1, dim=1)
		  equals = top_class == label.view(*top_class.shape)
		  correct += torch.mean(equals.type(torch.FloatTensor))

	    train_loss = train_loss/len(train_loader.dataset)
	    valid_loss = valid_loss/len(valid_loader.dataset)
	    if valid_loss < min_loss:
      		min_loss= valid_loss
      		save_model(model, model_name, train_dataset, optimizer, hidden)
	    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
	        epoch, train_loss, valid_loss, correct/len(valid_loader)))


def load_checkpoint(filename, lr):
	checkpoint= torch.load(filename)
	arch= checkpoint['arch']
	hidden_units= checkpoint['hidden_units']
	if arch== 'vgg16':
		model= models.vgg16(pretrained=True)
		for param in model.features.parameters():
			param.requires_grad= False
		model.classifier= nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
	                          nn.ReLU(),
	                          nn.Dropout(0.5),
	                          nn.Linear(checkpoint['hidden_units'], checkpoint['hidden_units']),
	                          nn.ReLU(),
	                          nn.Dropout(0.5),
	                          nn.Linear(checkpoint['hidden_units'], 102),
	                          nn.LogSoftmax(dim=1))
		optimizer= optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay= 0.0005, nesterov=True)
		criterion= nn.NLLLoss()

	elif arch == 'inception':
		model= models.inception_v3(pretrained=True)
		for param in model.parameters():
			param.requires_grad= False
		model.AuxLogits.fc = nn.Linear(768, 102)
		model.fc = nn.Linear(2048, 102)
		ct = []
		for name, child in inception.named_children():
    		if "Conv2d_4a_3x3" in ct:
	        	for params in child.parameters():
	            	params.requires_grad = True
    		ct.append(name)
    	optimizer= optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr= lr, momentum= 0.9)
    	criterion= nn.CrossEntropyLoss()
	scheduler= lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['opt_dict'])
	print('Checkpoint loaded successfully')
	return arch, model, hidden_units, optimizer, scheduler, criterion


def test_accuracy(model, test_loader, gpu):
	print('Printing model accuracy on test data:')
	correct=0
	train_on_gpu= gpu and torch.cuda.is_available()
	with torch.no_grad():
	for inputs, labels in test_loader:
	    if train_on_gpu:
	        inputs, labels = inputs.cuda(), labels.cuda()
	    outputs = model(inputs)
	    _, predicted = outputs.max(dim=1)
	    equals = predicted == labels.data
	    correct += equals.float().mean()
	print(correct/len(test_loader))


  if __name__ == "__main__":
  	text= 'This is a training script for classifying the Oxford 102 Flowers Dataset'
	parser= parser= argparse.ArgumentParser(description=text)
	parser.add_argument('mode', 
						help='', choice= ['train', 'resume_train'])
	parser.add_argument('dir', 
						help='root directory of image files')
	parser.add_argument('--checkpoint', help= 'checkpoint file')
	parser.add_argument('--model', 
						help='pretrained model',
						choice= ['vgg16', 'inception'], default= 'vgg16')
	parser.add_argument('--hidden_units', 
						help='number of hidden nodes',
						type=int, default= 4096)
	parser.add_argument('--epochs', 
						help='number of epochs to train',
						type=int, default= 30)
	parser.add_argument('--lr', 
						help='learning rate',
						type= float, default= 0.001)
	parser.add_argument('--GPU', 
						help='train with GPU, T/F',
						action='store_true', default=False)
	args= parser.parse_args()
	# init user input args
	use_mode= args.mode
	data_dir= args.dir
	init_model= args.model
	hidden= args.hidden_units
	n_epoch= args.epochs
	lr= args.lr
	gpu= args.GPU
	chkfile= args.checkpoint
	if use_mode=='resume_train' and chkfile==None:
		raise RunTimeError('No checkpoint file passed in')
	if use_mode=='train' and chkfile !=None:
		raise RunTimeError('Initial training should not have a checkpoint')

	if init_model== 'vgg16':
		resize_factor= 256
		crop_factor= 224
	elif init_model== 'inception':
		resize_factor= 512
		crop_factor= 299
	train_loader, valid_loader, test_loader, train_dataset= data_preprocess(data_dir, resize_factor, crop_factor)
	if use_mode== 'train':
		model= choose_model(init_model)
		if init_model=='vgg16':
			criterion= nn.NLLLoss()
			optimizer= optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay= 0.0005, nesterov=True)
		elif init_model== 'inception':
			criterion= nn.CrossEntropyLoss()
			optimizer= optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr= lr, momentum= 0.9)
		train(n_epoch, model, init_model, train_loader, valid_loader, criterion, optimizer, hidden, gpu)
		test_accuracy(model, test_loader, gpu)
	elif use_mode== 'resume_train':
		arch, model, hidden_units, optimizer, scheduler, criterion= load_checkpoint()
		train(n_epoch, model, arch, train_loader, valid_loader, criterion, optimizer, hidden_units, gpu)
		test_accuracy(model, test_loader, gpu)
