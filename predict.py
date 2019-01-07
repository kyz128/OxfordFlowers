#imports 
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import transforms, models, datasets
import json

def load_checkpoint(filename):
  checkpoint= torch.load(filename)
  arch= checkpoint['arch']
  hidden_units= checkpoint['hidden_units']
  model.class_to_idx= checkpoint['class_to_idx']
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
  model.load_state_dict(checkpoint['state_dict'])
  print('Checkpoint loaded successfully')
  return arch, model

def process_image(image, resize_factor, crop_factor):
    img= Image.open(image)
    img= img.resize(size=(resize_factor, resize_factor))
    center= (resize_factor-1)//2
    offset= crop_factor//2
    img= img.crop((center-offset, center-offset, center+offset, center+offset ))
    np_img= np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean)/std    
    np_img = np.transpose(np_img, (2, 0, 1))
    return np_img

def predict(image_path, checkpoint_file, topk, mapping_file, gpu):
    predict_on_gpu= gpu and torch.cuda.is_available()
    arch, model= load_checkpoint(checkpoint_file)
    model.cpu()
    with open(mapping_file, 'r') as f:
      cat_to_name = json.load(f)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    if arch=='vgg16':
      resize_factor= 256
      crop_factor= 224
    elif arch=='inception':
      resize_factor= 512
      crop_factor= 299
    img= process_image(image_path, resize_factor, crop_factor)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img= img.reshape((1, 3, crop_factor, crop_factor))
    if predict_on_gpu:
      model.cuda()
      img.cuda()
    model.eval()
    if arch=='vgg16':
      log_val = model(img)
    elif arch=='inception':
      initial= model(img)
      log_val= F.softmax(initial, dim=1) 
    prob_dist = torch.exp(log_val) 
    top_prob, top_class = prob_dist.topk(topk)
    top_prob= top_prob.detach().numpy()[0]
    top_class= top_class.detach().numpy()[0]
    if predict_on_gpu:
      top_class= top_class.cpu()
      top_prob= top_prob.cpu()
    top_fl_idx = [idx_to_class[clss] for clss in top_class]
    top_fl= [cat_to_name[f] for f in top_fl_idx]
    return top_prob, top_fl
    
  if __name__ == "__main__":
    text= 'This is the prediction script for classifying the Oxford 102 Flowers Dataset'
    parser= parser= argparse.ArgumentParser(description=text)
    parser.add_argument('img_file', help= 'pass in image for inference')
    parser.add_argument('checkpoint', help= 'checkpoint file')
    parser.add_argument('mapping', help= 'category to name mapping file')
    parser.add_argument('--GPU', help='train with GPU, T/F',
                        action='store_true', default=False)
    parser.add_argument('--topk', help='predict top k classes',
                        type=int, default=1)
    args= parser.parse_args()
    k= args.topk
    image= args.img_file
    checkpoint= args.checkpoint
    mapping_file= args.mapping 
    gpu= args.GPU
    predict(image, checkpoint, k, mapping_file, gpu)

