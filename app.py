from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from torch import nn
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from torchvision import transforms
import cv2
from torchvision import models
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
ind = 0
# Define a flask application
application = Flask(__name__)
device = 'cpu'
# Model saved with Keras model.save()
def prediction_bar(output,encoder):
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0]
    size = len(a)
    if(size>5):
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1*size:])
    prediction = list()
    clas = list()
    for i in a:
      prediction.append(float(output[:,i]*100))
      clas.append(str(i))
    cl = list()
    for i in a:
        cl.append(encoder[int(i)])
    plt.figure()
    plt.bar(cl,prediction)
    plt.title("Confidence score bar graph")
    plt.xlabel("Confidence score")
    plt.ylabel("Class number")
    plt.savefig('static/pred_bar.jpg')
def preprocess(path,test_transforms):
  img = cv2.imread(path)
  img = test_transforms(img)
  img = img.unsqueeze(0)
  return img
def im_convert(tensor,inv_normalize):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image= image.numpy()
    image = image.transpose(1,2,0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.clip(0, 1)
    return image
def cam(model,img,encoder,inv_normalize):
  fmap,logits = model(img)
  params = list(model.parameters())
  weight_softmax = model.linear.weight.detach().numpy()
  logits = sm(logits)
  idx = np.argmax(logits.detach().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap.detach().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  cam = out.reshape(h,w)
  cam = cam - np.min(cam)
  cam_img = cam / np.max(cam)
  cam_img = np.uint8(255*cam_img)
  img = im_convert(img,inv_normalize)
  h,w,c = img.shape
  out = cv2.resize(cam_img, (h,w))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  result = heatmap * 0.5 + img*0.8*255
  cv2.imwrite('static/cam.png',result)
  return logits
sm = nn.Softmax()

model_pneumonia = 'models/pneumonia.h5'
classes_pneumonia = ['NORMAL', 'PNEUMONIA']

decoder_pneumonia = {}
for i in range(len(classes_pneumonia)):
    decoder_pneumonia[classes_pneumonia[i]] = i
encoder_pneumonia = {}
for i in range(len(classes_pneumonia)):
    encoder_pneumonia[i] = classes_pneumonia[i]

class classifier_pneumonia(nn.Module):
    def __init__(self):
        super(classifier_pneumonia, self).__init__()
        model = models.densenet121(pretrained = False)
        model = model.features
        self.model = model
        self.linear = nn.Linear(1024, 2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.6)
    def forward(self, input):
        am = self.model(input)
        out = nn.functional.adaptive_avg_pool2d(am, output_size = 1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        res = self.linear(self.dropout2(self.bn1(out)))
        return am,res
net_pneumonia = classifier_pneumonia().to(device)
net_pneumonia.load_state_dict(torch.load(model_pneumonia,map_location=lambda storage, loc: storage))
net_pneumonia.eval()

mean_pneumonia = torch.from_numpy(np.asarray([0.4823, 0.4823, 0.4823]))
std_pneumonia = torch.from_numpy(np.asarray([0.2218, 0.2218, 0.2218]))
transform_pneumonia = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_pneumonia,std_pneumonia)])

inv_normalize_pneumonia = inv_normalize =  transforms.Normalize(
    mean=-1*np.divide(mean_pneumonia,std_pneumonia),
    std=1/std_pneumonia
)

def preprocess_pneumonia(image_path):
    image = cv2.imread(image_path)
    image = transform_pneumonia(image)
    image = image.view((1,3,224,224))
    return image
import string
import random
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def pneumonia_predict(image_path):
    image = preprocess_pneumonia(image_path)
    image = image.type(torch.FloatTensor)
    pred = cam(net_pneumonia,image,encoder_pneumonia,inv_normalize_pneumonia)
    prediction_bar(pred,encoder_pneumonia)
    img = cv2.imread('static/pred_bar.jpg')
    img = cv2.resize(img,(300,224))
    img2 = cv2.imread('static/cam.png')
    image = cv2.imread(image_path)
    image = cv2.resize(image,(224,224))
    img1 = np.append(img,img2,axis = 1)
    txt = randomString()
    cv2.imwrite('static/{}.jpg'.format(txt),img1)
    pred1,pred = torch.max(pred,dim = 1)
    if(os.path.exists('static/results.csv')):
        data = pd.read_csv('static/results.csv')
        print(txt)
        a = [{'file_name':txt,'prediction':pred.item(),'confidence':pred1.item()}] 
        data = data.append(pd.DataFrame(a))
        print(data)
        data.to_csv('./static/results.csv',index = False)
    else:
        print(pred)
        data = [{'file_name':txt,'prediction':pred.item(),'confidence':pred1.item()}]
        df = pd.DataFrame(data)
        df.to_csv('./static/results.csv',index= False)
    return pred,txt



@application.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')    

@application.route('/pneumonia', methods=['GET'])
def pneumonia():
    # Main page
    return render_template('xxx.html')

@application.route('/instructions', methods=['GET'])
def instructions():
    # Main page
    return render_template('instructions.html')


@application.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
        basepath = os.getcwd()
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pred,ind = pneumonia_predict(file_path)
        return './static/' + ind + '.jpg'
    return None

if __name__ == '__main__':
    application.run(port = 90)

    
