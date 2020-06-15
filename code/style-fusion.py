from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))).float().to(device)
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

def transform_back_image(img):
    #img is a tensor of size [B, C, H, W]
    img = img.data.to(device)
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    img = img * std + mean
    img = img.clamp(0, 1)[0,:,:,:]
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255.0
    return img.astype(np.uint8)

#载入图片
content_input = cv2.imread("content.jpg")
content = cv2.resize(content_input, (256, 256))
content_tensor = transform_image(content)

styleA_input = cv2.imread("style1.jpg")
styleA = cv2.resize(styleA_input, (256, 256))
styleA_tensor = transform_image(styleA)

styleB_input = cv2.imread("style2.jpg")
styleB = cv2.resize(styleB_input, (256, 256))
styleB_tensor = transform_image(styleB)

lamda = 0.2 #lamda 在[0,1]间取值，表示风格A的占比

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

vgg.to(device)

def Features(image, model, layers = None):
    #image as a tensor
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def Gram(feature):
    n, c, h, w = feature.size()
    feature = feature.view(c, h * w)
    gram = torch.mm(feature, feature.t())
    
    return gram 

content_features = Features(content_tensor, vgg)
styleA_features = Features(styleA_tensor, vgg)
styleA_grams = {layer: Gram(styleA_features[layer]) for layer in styleA_features}

styleB_features = Features(styleB_tensor, vgg)
styleB_grams = {layer: Gram(styleB_features[layer]) for layer in styleB_features}

#target_input = content

target_input = cv2.imread("content.jpg")
target_input = cv2.resize(target_input, (256, 256))
target = transform_image(target_input).requires_grad_(True)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

alpha = 1e6  # content_weight
beta = 5e6  # style_weight

show_every = 10
optimizer = optim.Adam([target], lr=0.5)
steps = 5000

def StyleLoss(target_features, style_grams, style_weights):
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = Gram(target_feature)
        n, c, h, w = target_feature.shape

        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (c * h * w)
    
    return style_loss 

for epo in range(1, steps+1):
    target_features = Features(target, vgg)

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    styleA_loss = StyleLoss(target_features, styleA_grams, style_weights)
    styleB_loss = StyleLoss(target_features, styleB_grams, style_weights)
    style_loss = styleA_loss * lamda + styleB_loss * (1 - lamda)
        
    total_loss = alpha * content_loss + beta * style_loss
    
    # update my target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(epo, 'Total loss: ', total_loss.item())
    # display intermediate images and print the loss
    if  epo % show_every == 0:
        #print('Total loss: ', total_loss.item())

        result = transform_back_image(target)
        cv2.imwrite("result.png", result)

result = transform_back_image(target)
cv2.imwrite("result.png", result)
    