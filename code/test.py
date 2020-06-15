import torch
import cv2
from model import Model
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def DeNormalize(tensor, device):
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)  # 返回一个新的张量，有上下限
    return res


def ReSize0(img):
    W = img.size[0]
    H = img.size[1]
    # 3/4 = 0.75    1    4/3 = 1.333
    if W / H < 0.875:
        img = img.resize((225,300))
    else :
        if W / H < 1.1667:
            img = img.resize((300,300))
        else:
            img = img.resize((300,225))
    return img

def ReSize1(img):
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



def isGray(img):
    W, H = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sum = 0
    for i in range(0, W):
        for j in range(0, H):
            sum = sum + s[i,j]
    print(sum / (W * H))
    if (sum / (W * H) < 15):
        return 1
    else:
        return 0


def cal_main(orginal_file, style_file):
    print("in")
    content_path = orginal_file
    style_path = style_file
    alpha = 1  # 风格混合度 可调
    device = 'cpu'  # 实际上可能可以是GPU(如果装了的话)

    model = Model()  # 实例化模型
    model.load_state_dict(torch.load('model_state.pth', map_location=lambda storage, loc: storage))  # 载入预先训练好的模型
    model = model.to(device)

    # 载入图片
    Normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化，起到稳定训练的作用
    trans = transforms.Compose([transforms.ToTensor(), Normalize])  # 格式转换与归一化

    content_image = Image.open(content_path)
    content_image = ReSize1(content_image)
    content_tensor = trans(content_image).unsqueeze(0).to(device)

    style_image = Image.open(style_path)
    style_image = ReSize1(style_image)
    style_tensor = trans(style_image).unsqueeze(0).to(device)

    print("input")
    # 生成目标图像
    with torch.no_grad():
        result = model.generate(content_tensor, style_tensor, alpha)


    result = DeNormalize(result, device) #去归一化，让它的着色更自然

    save_image(result, 'out.jpg')

    img = Image.open('out.jpg')
    style_image = cv2.imread(style_path)

    if isGray(style_image):
        img = img.convert("L")

    img.save('result.jpg')
