import matplotlib.pyplot as plt
import cv2
import sys

img = cv2.imread('./result1.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
W,H = h.shape[:2]

isok = h
for i in range(0, W):
    for j in range(0, H):
        if (s[i,j] >= 10 and v[i,j] > 200):
            isok[i,j] = 0
        else :
            isok[i,j] = 1

'''
bad = img
for i in range(0, W):
    for j in range(0, H):
        if (isok[i,j] == 0): #取出我们认为的绿点或者具有某些属性的点，这里的条件可以调整
            bad[i,j,0] = bad[i,j,1] = bad[i,j,2] = 0
        else:
            bad[i,j,0] = bad[i,j,1] = bad[i,j,2] = 255
   
cv2.imwrite('bad.jpg', bad)  
'''
#取邻域内好点的均值填充
n = 512
imgM = img
for i in range(0, n):
    for j in range(0, n):
            if (isok[i,j] == 0):
                cnt = 0 
                s0 = 0
                s1 = 0
                s2 = 0
                for dx in range(-8,8):
                    for dy in range(-8,8):
                        nx = i + dx
                        ny = j + dy
                        if nx >= 0 and nx < n and ny >= 0 and ny < n and isok[nx,ny] == 1:
                            cnt = cnt + 1 
                            s0 = img[nx,ny,0] + s0
                            s1 = img[nx,ny,1] + s1
                            s2 = img[nx,ny,2] + s2
                imgM[i,j,0] = s0 / cnt
                imgM[i,j,1] = s1 / cnt
                imgM[i,j,2] = s2 / cnt
                
cv2.imwrite('filling1.jpg', imgM)

'''
#取最近点填充的部分
queue = []
for i in range(0, n):
    for j in range(0, n):
            if (isok[i,j]):
                queue.append([i,j])
dires = [[0, 1], [0, -1], [1, 0], [-1, 0]]
while queue:
    x, y = queue.pop(0)
    for dx, dy in dires:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and isok[nx][ny] == 0:
            imgM[nx, ny,0] = img[x, y, 0]
            imgM[nx, ny,1] = img[x, y, 1]
            imgM[nx, ny,2] = img[x, y, 2]
            queue.append([nx, ny])
            isok[nx][ny] = 1

cv2.imwrite('filling2.jpg', imgM)
'''