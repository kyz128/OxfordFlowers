# OxfordFlowers
Image Classifier on 102 Oxford Flowers Dataset

Compile train.py for the training script and predict.py for the prediction script.
Both are command line applications that run automatically upon compilation. 
Here are the following arguments each script takes:train.py

mode: either train or resume_train
dir: directory of image files for training 
checkpoint: path to the checkpoint file
model: default is vgg16
hidden_units: default is 4096
epochs: default is 30
lr: learning rate; default is 0.001
GPU: use GPU for training; default is false

predict.py

img_file: pass in image for inference
checkpoint: checkpoint file
mapping: category to name mapping file
GPU: use GPU for prediction; default is false
topk: predict top k classes; default is 1

See Jupyter Notebook for output. 
                        

