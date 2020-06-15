import torch
import cv2
from model import Model
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

img = cv2.imread("./filling1.jpg")
img2 = img
img2 = cv2.medianBlur(img2, 3)
img2 = cv2.medianBlur(img2, 3) 

cv2.imwrite('./smooth2.png', img2)
img = Image.open('smooth2.png')
gray = img.convert("L")
gray.save('final_out.jpg')