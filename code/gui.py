# -*- coding: utf-8 -*-
# 创建GUI窗口打开图像 并显示在窗口中

import PIL
from PIL import ImageTk  # 导入图像处理函数库
import tkinter as tk  # 导入GUI界面函数库
from tkinter import *
import tkinter.filedialog
# from getJPG import *
from test import *
# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('图像显示界面')
window.geometry('1220x400')
global img_png  # 定义全局变量 图像的
# var = tk.StringVar()    # 这时文字变量储存器

flag1 = False
flag2 = False
orginal_pic = "1.jpg"
style_pic = "2.jpg"

def ReSize(img):
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


Img_p = PIL.Image.open("空.png")
new_p = Img_p.resize((110, 110), PIL.Image.BILINEAR)
img_plus = ImageTk.PhotoImage(new_p)
plus_label = tk.Label(window, image=img_plus)
plus_label.grid(row = 2, column = 1)
new_whi = Img_p.resize((450, 300), PIL.Image.BILINEAR)
new_white = Img_p.resize((300, 300), PIL.Image.BILINEAR)



# 创建打开图像和显示图像函数
def Open_Img():
    global img_png
    global orginal_pic
    global img_plus
    global flag1
    global flag2
    # var.set('已打开')
    flag1 = True
    filename = tkinter.filedialog.askopenfilename()
    orginal_pic = filename
    # print(filename)
    # orginal_pic = "test.jpg"
    Img = PIL.Image.open(orginal_pic)
    # Img = PIL.Image.open(e.get())
    #newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    newImg = ReSize(Img)
    img_png = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png)
    label_Img.grid(row=2, column=0, columnspan=1, rowspan=2)
    Img_p = PIL.Image.open("plus.jpg")
    new_p = Img_p.resize((100, 100), PIL.Image.BILINEAR)
    img_plus = ImageTk.PhotoImage(new_p)
    plus_label = tk.Label(window, image=img_plus)
    plus_label.grid(row = 2, column = 1)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)


Img_0 = PIL.Image.open("打开.jpg")
new_0 = Img_0.resize((310, 40), PIL.Image.BILINEAR)
img_p0 = ImageTk.PhotoImage(new_0)

# 创建打开图像按钮
btn_Open = tk.Button(window,
                     text='打开图像',  # 显示在按钮上的文字
                     image = img_p0,compound="center"
                     , command=Open_Img)  # 点击按钮式执行的命令
btn_Open.grid(row=0, column=0,rowspan = 2)  # 按钮位置


# 风格1
def sty1_open():
    global img_png1
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style5.jpg"
    Img = PIL.Image.open(r"style\style5.jpg")
    newImg = ReSize(Img)
    #newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png1 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png1)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_1 = PIL.Image.open(r"style\style5.jpg")
new_1 = Img_1.resize((73, 20), PIL.Image.BILINEAR)
img_p1 = ImageTk.PhotoImage(new_1)


sty1 = tk.Button(window,
                 text='水墨',  # 显示在按钮上的文字
                 image = img_p1,compound="center", command=sty1_open)  # 点击按钮式执行的命令
sty1.grid(row=1, column=2)

# 风格20
def sty02_open():
    global img_png02
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style0.jpg"
    Img = PIL.Image.open(r"style\style0.jpg")
    newImg = ReSize(Img)
    #ewImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png02 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png02)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_02 = PIL.Image.open(r"style\style0.jpg")
new_02 = Img_02.resize((73, 20), PIL.Image.BILINEAR)
img_p02 = ImageTk.PhotoImage(new_02)


sty02 = tk.Button(window,
                 text='抽象',  # 显示在按钮上的文字
                 image = img_p02,compound="center", command=sty02_open)  # 点击按钮式执行的命令
sty02.grid(row=0, column=2)

# 风格2
def sty2_open():
    global img_png2
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style2.jpg"
    Img = PIL.Image.open(r"style\style2.jpg")
    newImg = ReSize(Img)
    #newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png2 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png2)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_2 = PIL.Image.open(r"style\style2.jpg")
new_2 = Img_2.resize((73, 20), PIL.Image.BILINEAR)
img_p2 = ImageTk.PhotoImage(new_2)


sty2 = tk.Button(window,
                 text='水彩',  # 显示在按钮上的文字
                 image = img_p2,compound="center", command=sty2_open)  # 点击按钮式执行的命令
sty2.grid(row=0, column=3)


# 风格30
def sty03_open():
    global img_png03
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style1.jpg"
    Img = PIL.Image.open(r"style\style1.jpg")
    newImg = ReSize(Img)
   # newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png03 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png03)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_03 = PIL.Image.open(r"style\style1.jpg")
new_03 = Img_03.resize((73, 20), PIL.Image.BILINEAR)
img_p03 = ImageTk.PhotoImage(new_03)


sty03 = tk.Button(window,
                 text='素描',  # 显示在按钮上的文字
                 image = img_p03,compound="center", command=sty03_open)  # 点击按钮式执行的命令
sty03.grid(row=1, column=3)


# 风格3
def sty3_open():
    global img_png3
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    filename = tkinter.filedialog.askopenfilename()
    style_pic = filename
    Img = PIL.Image.open(filename)
    newImg = ReSize(Img)
    #newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png3 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png3)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_3 = PIL.Image.open("b3.jpg")
new_3 = Img_3.resize((73, 20), PIL.Image.BILINEAR)
img_p3 = ImageTk.PhotoImage(new_3)

sty3 = tk.Button(window,
                 text='浏览',  # 显示在按钮上的文字
                 image = img_p3,compound="center", command=sty3_open)  # 点击按钮式执行的命令
sty3.grid(row=0, column=5)


# 风格40
def sty04_open():
    global img_png04
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style3.jpg"
    Img = PIL.Image.open(r"style\style3.jpg")
    newImg = ReSize(Img)
    #newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png04 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png04)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_04 = PIL.Image.open(r"style\style3.jpg")
new_04 = Img_04.resize((73, 20), PIL.Image.BILINEAR)
img_p04 = ImageTk.PhotoImage(new_04)


sty04 = tk.Button(window,
                 text='油彩',  # 显示在按钮上的文字
                 image = img_p04,compound="center", command=sty04_open)  # 点击按钮式执行的命令
sty04.grid(row=0, column=4)

# 风格50
def sty05_open():
    global img_png05
    global style_pic
    global flag1
    global flag2
    # var.set('已打开')
    # filename = tkinter.filedialog.askopenfilename()
    # style_pic = filename
    # print(filename)
    flag2 = True
    img_blank1 = ImageTk.PhotoImage(new_white)
    blank_label1 = tk.Label(window, image = img_blank1)
    blank_label1.grid(row = 2, column = 2, rowspan = 2, columnspan = 4)
    style_pic = r"style\style4.jpg"
    Img = PIL.Image.open(r"style\style4.jpg")
    newImg = ReSize(Img)
   # newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
    img_png05 = ImageTk.PhotoImage(newImg)
    label_Img = tk.Label(window, image=img_png05)
    label_Img.grid(row=2, column=2, columnspan=4, rowspan=2)
    img_blank = ImageTk.PhotoImage(new_whi)
    blank_label = tk.Label(window, image = img_blank)
    blank_label.grid(row = 2, column = 6, rowspan = 2, columnspan = 4)

Img_05 = PIL.Image.open(r"style\style4.jpg")
new_05 = Img_05.resize((73, 20), PIL.Image.BILINEAR)
img_p05 = ImageTk.PhotoImage(new_05)


sty05 = tk.Button(window,
                 text='报纸',  # 显示在按钮上的文字
                 image = img_p05,compound="center", command=sty05_open)  # 点击按钮式执行的命令
sty05.grid(row=1, column=4)


def change_style():
    global orginal_pic
    global style_pic
    global img_png4
    global img_elus
    global flag1
    global flag2
    if flag1 and flag2:
        cal_main(orginal_pic, style_pic)
        Img_p = PIL.Image.open("equal.jpg")
        new_p = Img_p.resize((110, 110), PIL.Image.BILINEAR)
        img_elus = ImageTk.PhotoImage(new_p)
        plus_label = tk.Label(window, image=img_elus)
        plus_label.grid(row = 2, column = 6)
        Img = PIL.Image.open("result.jpg")
        newImg = ReSize(Img)
   #     newImg = Img.resize((300, 300), PIL.Image.BILINEAR)
        img_png4 = ImageTk.PhotoImage(newImg)
        label_Img = tk.Label(window, image=img_png4)
        label_Img.grid(row=2, column=7, columnspan=3, rowspan=2)
    
    pass


change = tk.Button(window,
                   text='转换',  # 上传完原始图片和风格图片后，点击转换
                   width=10, height=1, command=change_style)  # 点击按钮式执行的命令
change.grid(row=1, column=5)

# 运行整体窗口
window.mainloop()
