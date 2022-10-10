import cv2
import os
import numpy as np

model=cv2.createBackgroundSubtractorMOG2()


def norm(img):
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    return img


ref_img_sum=cv2.imread('E:/python/aobishixi/study/imgs/imgs/0.png', -1)
files = os.listdir('E:/python/aobishixi/study/imgs/imgs')
files.sort(key=lambda x: int(x.split('.')[0]))
for filename in files:
    ref_img = cv2.imread('E:/python/aobishixi/study/imgs/imgs'+'/'+filename,-1)
    ref_img_sum = cv2.addWeighted(ref_img,0.5,ref_img_sum,0.5,0, dtype = cv2.CV_32F)
    if filename == "144.png":
        break


def check_path(pathname,pathname1,ref_img):
    files = os.listdir(pathname)
    files.sort(key=lambda x: int(x.split('.')[0]))
    for filename in files:
        print(filename)
        cur_img = cv2.imread(pathname+'/'+filename,-1)
        #做帧差
        diff_frame = cv2.absdiff(cur_img, ref_img)

        #选取区域
        diff_frame_roi = diff_frame[200:, 200:540]

        #图转二值图
        frame_bin = cv2.inRange(diff_frame_roi,20 ,100)

        #闭运算
        kernel = np.ones((3,3) , dtype=np.uint8)
        closing = cv2.morphologyEx(frame_bin,cv2.MORPH_CLOSE, kernel)

        #寻找目标
        contours, _ = cv2.findContours(closing,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)

        x_obj, y_obj = 0, 0
        w, h = 0, 0
        xl, yl = [], []
        wl, hl = [], []
        for cont in contours:
            (x, y), radius = cv2.minEnclosingCircle(cont)
            if radius > 13 and radius < 28:
                rect = cv2.minAreaRect(cont)
                x_obj, y_obj = rect[0]
                w, h = rect[1]
                if w/h > 1.5 or w/h < 0.5 or w*h < 500:
                    continue
                else:
                    xl.append(x_obj)
                    yl.append(y_obj)
                    wl.append(w)
                    hl.append(h)

        x_final = [xi + 200 for xi in xl]
        y_final = [yi + 200 for yi in yl]
        for x_f, y_f, w_f, h_f in zip(x_final, y_final, wl, hl):
            cv2.rectangle(cur_img, (int(x_f - w_f / 2), int(y_f - h_f / 2)), (int(x_f + w_f / 2), int(y_f + h_f / 2)), (0, 255, 0), thickness=2)       

        img1=norm(cur_img)
        cv2.imwrite(pathname1+'/'+filename,img1)

check_path("E:/python/aobishixi/study/imgs/imgs","E:/python/aobishixi/study/imgs/img2s",ref_img)