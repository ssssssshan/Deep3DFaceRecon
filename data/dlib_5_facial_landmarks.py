import os
import numpy as np
import cv2
import dlib
import time

# 加载训练模型
PREDICTOR_PATH = "../checkpoints/lm_model/shape_predictor_5_face_landmarks.dat"
# dlib接口提取面部标记
# 将一个图像转化成numpy数组，并返回一个5x2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# order = [40, 37, 43, 46, 34, 49, 55]  # 左内,左外,右内,右外,鼻子,左嘴,右嘴
file_dir = "../datasets/test1/"
save_dir = file_dir + "detections/"
# saveimg_dir = file_dir+"detections_img/"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("mkdir detections")
# if not os.path.exists(saveimg_dir):
#     os.mkdir(saveimg_dir)
#     print("mkdir detections_img")

sum = 0
pics = 0
for file in os.listdir(file_dir):
    if file.endswith('png') or file.endswith('jpg'):
        sum += 1
        fname = file_dir + file
        start = time.time()
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = detector(img_gray, 1)
        end = time.time()

        if len(rects) != 1:
            print('*******detection {} failed********'.format(file))
        else:

            landmarks = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[0]).parts()])
            save_fname = save_dir + file.split('.')[0] + '.txt'

            with open(save_fname, "w") as f:
                for i in range(0, landmarks.shape[0]):
                    for j in range(landmarks.shape[1]):
                        f.write(str(landmarks[i, j]) + '\t')
                    f.write('\n')
            f.close()
            print('detection {} complete in {:.0f}m {:.3f}s'.format(file, (end - start) // 60, (end - start) % 60))
            pics += 1

            # for idx, point in enumerate(landmarks):
            #     cv2.circle(img, (point[0, 0], point[0, 1]), 2, (0, 255, 0), 2)
            # cv2.imwrite(os.path.join(saveimg_dir, file), img)

print('save {}/{} pictures'.format(pics, sum))
