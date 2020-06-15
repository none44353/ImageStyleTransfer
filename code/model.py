import torch.nn as nn
from torchvision.models import vgg19

def calc_Mean_Std(features): #计算特征向量的平均值和标准差
    #features是[n,c,h,w]规模的，我们希望返回feature的关于batch中每一张图每一个通道的平均值，标准差
    n, c = features.size()[:2] #feature前两个维度的大小就是单个batch中的图像数n,通道数c
    mean = features.reshape(n, c, -1).mean(dim = 2).reshape(n, c, 1, 1)
    std = features.reshape(n, c, -1).std(dim = 2).reshape(n, c, 1, 1) + 1e-6 #一个小量，防止做除法的时候除数太小
    return mean, std

def AdaIN(content_features, style_features): #对特征向量作变换，得到内容为content，风格为style的目标图的特征向量
    #features是[n,c,h,w]规模的
    content_mean, content_std = calc_Mean_Std(content_features)
    style_mean, style_std = calc_Mean_Std(style_features)
    goal_features = style_std * (content_features - content_mean) / content_std + style_mean
    return goal_features

#利用VGG19的前若干层构encoder和decoder
class Encoder(nn.Module):
    def __init__(self):#利用预先训练好的网络定义感知损失
        super().__init__()
        vgg = vgg19(pretrained = True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():#在训练过程中保持encoder的参数不变
            p.requires_grad = False

    def forward(self, images, output_last_feature = False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4

class RC(nn.Module): #定义一个卷积层（可能可以不使用激活函数）
    def __init__(self, in_channels, out_channels, kernel_size = 3, pad_size = 1, activated = True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size)) #填充边界 left, right, top, bottom  图是H*W的
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size) #做卷积，这样保证图的大小不变
        self.activated = activated #是否有激活层

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return nn.functional.relu(h)
        else:
            return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = nn.functional.interpolate(h, scale_factor=2) #上采样，插值，对应maxpool
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = nn.functional.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = nn.functional.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = Encoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha = 1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature = True)
        style_features = self.vgg_encoder(style_images, output_last_feature = True)
        t = AdaIN(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features #混合度决定原图内容的占比
        out = self.decoder(t)
        return out

    @staticmethod #静态方法 名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包
    def calc_content_loss(out_features, t):
        return nn.functional.mse_loss(out_features, t) #用特征向量的均方损失函数衡量内容的相似度（差的内积）

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_Mean_Std(c)
            s_mean, s_std = calc_Mean_Std(s)
            loss += nn.functional.mse_loss(c_mean, s_mean) + nn.functional.mse_loss(c_std, s_std) #用均值、标准差的差距衡量风格相似度 per picture not per batch

        return loss

    def forward(self, content_images, style_images, alpha = 1.0, lam = 10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = AdaIN(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, t)#认为卷积网络的深层反馈更多表示图像的内容
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)#卷积网络的中间层反馈更多表示图像的风格
        loss = loss_c + lam * loss_s
        return loss
