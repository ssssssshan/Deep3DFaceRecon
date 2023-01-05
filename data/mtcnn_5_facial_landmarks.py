import cv2
import os
import numpy as np
from mtcnn import MTCNN
import time

file_dir = "../datasets/test2/"
save_dir = file_dir+"detections/"
# saveimg_dir = file_dir+"detections_img/"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("mkdir detections")
# if not os.path.exists(saveimg_dir):
#     os.mkdir(saveimg_dir)
#     print("mkdir detections_img")

detector=MTCNN()
keypoints=['left_eye', 'right_eye','nose', 'mouth_left', 'mouth_right']
sum=0
pics=0
for file in os.listdir(file_dir):
    if file.endswith('png') or file.endswith('jpg'):
        fname = file_dir+file
        start = time.time()
        image=cv2.imread(fname)
        result=detector.detect_faces(image)
        end = time.time()
        sum+=1

        if len(result)==1:
            landmarks=result[0]['keypoints']

            save_fname = save_dir + file.split('.')[0] + '.txt'
            with open(save_fname, "w") as f:
                for i in range(5):
                    f.write(str(landmarks[keypoints[i]][0]) + '\t' + str(landmarks[keypoints[i]][1]))
                    f.write('\n')
            f.close()

            print('detection {} complete in {:.0f}m {:.3f}s'.format(file, (end-start)// 60, (end-start) % 60))
            pics+=1

            # bounding_box = result[0]['box']
            # x1 = bounding_box[0]
            # y1 = bounding_box[1]
            # x2 = bounding_box[0] + bounding_box[2]
            # y2 = bounding_box[1] + bounding_box[3]
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # for i in range(len(keypoints)):
            #     cv2.circle(image, landmarks[keypoints[i]], 2, (0, 255, 0), 2)
            # cv2.imwrite(os.path.join(saveimg_dir, file), image)
        else:
            print('*******detection {} failed********'.format(file))
print('save {}/{} pictures'.format(pics,sum))