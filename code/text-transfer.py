import torch
import cv2
from model import Model
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def ReSize(img):
    W = img.size[0]
    H = img.size[1]
    # 3/4 = 0.75    1    4/3 = 1.333
    if W / H < 0.875:
        img = img.resize((375,500))
    else :
        if W / H < 1.1667:
            img = img.resize((500,500))
        else:
            img = img.resize((500,375))
    return img

out = cv2.imread('./out.jpg')
out_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(out_hsv)
H, W = h.shape[:2]

content_path = './content.jpg'
content_image = Image.open(content_path)
content_image = content_image.resize((W, H))
content_image.save('color.jpg')

color = cv2.imread('./color.jpg')
color_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

print(color_hsv.shape)
print(out_hsv.shape)

for i in range(0, H):
    for j in range(0, W):
        out_hsv[i,j, 0] = color_hsv[i,j, 0]
        out_hsv[i,j, 1] = color_hsv[i,j, 1]
out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('result.jpg', out)

