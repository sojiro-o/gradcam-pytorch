import torch
import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
import torch.nn as nn
import glob



resnet = models.resnet18(num_classes=8)
resnet.load_state_dict(torch.load("./model_save/best.pth")["model_state_dict"])


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            #print('name=',name)
            #print('x.size()=',x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers,use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        #print('classfier=',output.size())
        if self.cuda:
            output = output.cpu()
            output = resnet.fc(output).cuda()
        else:
            output = resnet.fc(output)
        return target_activations, output

def preprocess_image(img):
    preprocessed_img = img.copy()[: , :, ::-1]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img
    input.requires_grad = True
    return input

def show_cam_on_image(img, mask, image_name):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # name = image_name.split(".jpg")[0] + "_cam.jpg"
    os.makedirs("./results_img", exist_ok=True)
    cv2.imwrite("./results_img/{}".format(image_name), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
        #self.model.zero_grad()
        one_hot.backward(retain_graph=True)##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #print('grads_val',grads_val.shape)
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        #print('weights',weights.shape)
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        #print('cam',cam.shape)
        #print('features',features[-1].shape)
        #print('target',target.shape)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (320, 320))
        # cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam



class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model#这里同理，要的是一个完整的网络，不然最后维度会不匹配。
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        for module in self.model.named_modules():
            module[1].register_backward_hook(self.bp_relu)

    def bp_relu(self, module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)
    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        #print(input.grad)
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad = True
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        #self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-dir', type=str, default='./test_img',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    model = models.resnet18(num_classes=8)
    model.load_state_dict(torch.load("./model_save/best.pth")["model_state_dict"])
    # print(model)
    del model.fc

    grad_cam = GradCam(model , \
                    target_layer_names = ["layer4"], use_cuda=args.use_cuda)

    # image_path = "/ML/datasets/car-rot/train/4/002200.jpg"
    # print(glob.glob("{}/*".format(args.image_dir)))
    for image_path in glob.glob("{}/*".format(args.image_dir)):
        print(image_path)
        image = cv2.imread(image_path,1)
        image_name = image_path.split("/")[-1]
        image = np.float32(cv2.resize(image, (320, 320))) / 255
        input = preprocess_image(image)
        input.required_grad = True
        # print('input.size()=',input.size())
        target_index =None

        mask = grad_cam(input, target_index)
        show_cam_on_image(image, mask, image_name)
