import os

os.environ["PATH"] = os.environ["PATH"] + ":/opt/conda/bin/ninja"
from options.test_options import TestOptions
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
from scipy.io import loadmat
import time
import cv2
from mtcnn import MTCNN
import tensorflow as tf
import dlib

mean_face = np.loadtxt('util/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])

focal = 1015.
center = 112.

def makedirs(name):
    save_path1 = os.path.join(name, '2Dlandmarks')
    save_path2 = os.path.join(name, '3DPoints')
    save_path3 = os.path.join(name, '3Dlandmarks')
    save_path4 = os.path.join(name, 'Pose')
    save_path5 = os.path.join(name, '2Dto3Dlandmarks')
    save_path6 = os.path.join(name, '3Dto2Dlandmarks')

    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
        print("mkdir 2D 68 landmarks")
    if not os.path.exists(save_path2):
        os.mkdir(save_path2)
        print("mkdir 3DPoints")
    if not os.path.exists(save_path3):
        os.mkdir(save_path3)
        print("mkdir 3D 68 landmarks")
    if not os.path.exists(save_path4):
        os.mkdir(save_path4)
        print("mkdir Pose")
    if not os.path.exists(save_path5):
        os.mkdir(save_path5)
        print("mkdir 2Dto3D landmarks")
    if not os.path.exists(save_path6):
        os.mkdir(save_path6)
        print("mkdir 3Dto2D landmarks")

    return save_path1,save_path2,save_path3,save_path4,save_path5,save_path6

def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    print("size：({}, {})".format(W,H))
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    # draw_landmarks(np.array(im)[:, :, ::-1], lm, '1.jpg', [0, 0, 255])
    _, im, lm, _ = align_img(im, lm, lm3d_std)  # 裁剪、对齐
    # draw_landmarks(np.array(im)[:,:,::-1], lm, '1.jpg', [0, 0, 255])
    if to_tensor:
        im_t = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm_t = torch.tensor(lm).unsqueeze(0)
    return im, lm, im_t, lm_t

def get_5_landmarks(file_dir):
    save_dir = os.path.join(file_dir,"detections")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("mkdir 5 detections")

    im_path, lm_path = get_data_path(file_dir)
    detector = MTCNN()

    for i in range(len(im_path)):
        print('Detect landmarks:', i, im_path[i])
        start = time.time()
        image = cv2.imread(im_path[i])
        if not os.path.isfile(lm_path[i]):
            detected_faces = detector.detect_faces(image)
            if len(detected_faces) == 1:
                landmarks=detected_faces[0]['keypoints'].values()
                with open(lm_path[i], "w") as f:
                    for keypoint in landmarks:
                        f.write(str(keypoint[0]) + '\t' + str(keypoint[1]))
                        f.write('\n')
                f.close()
            else:
                print('*******detection {} failed********'.format(im_path[i]))
        end = time.time()
        print('Img 5 landmarks detection complete in {:.3f}s'.format(end - start))
    print('Detect', file_dir, "completed")

def load_lm_graph(graph_filename):
    with tf.io.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        lm_sess = tf.compat.v1.Session(graph=graph)

    return lm_sess,img_224,output_lm

def load_data(img_name, txt_name):
    return cv2.imread(img_name), np.loadtxt(txt_name)

def draw_landmarks(img, landmark, save_name, color):
    landmark = landmark
    lm_img = np.zeros([img.shape[0], img.shape[1], 3])
    lm_img[:] = img.astype(np.float32)
    landmark = np.round(landmark).astype(np.int32)

    for i in range(len(landmark)):
        for j in range(-1, 1):
            for k in range(-1, 1):
                if img.shape[0] - 1 - landmark[i, 1]+j > 0 and \
                        img.shape[0] - 1 - landmark[i, 1]+j < img.shape[0] and \
                        landmark[i, 0]+k > 0 and \
                        landmark[i, 0]+k < img.shape[1]:
                    lm_img[img.shape[0] - 1 - landmark[i, 1]+j, landmark[i, 0]+k,
                           :] = np.array(color)

    lm_img = lm_img.astype(np.uint8)
    cv2.imwrite(save_name, lm_img)

def get_2D68_landmarks(name, img_name, img, sess, input_op, output_op):
    input_img_ = np.array(img)[:, :, ::-1]
    input_img = np.reshape(input_img_, [1, 224, 224, 3]).astype(np.float32)
    landmark = sess.run(output_op, feed_dict={input_op: input_img})
    landmark = landmark.reshape([68, 2]) + mean_face

    draw_landmarks(input_img[0], landmark, os.path.join(save_path, img_name), [0, 0, 255])
    np.savetxt(os.path.join(save_path, img_name.split('.')[0]+'.txt'), landmark)

    return os.path.join(save_path, img_name), os.path.join(save_path, img_name.split('.')[0]+'.txt')

def get_2D68_landmarks_dlib(save_path, img_name, img, detector, predictor):
    input_img = np.array(img)[:, :, ::-1]
    rects = detector(input_img, 1)
    landmark = np.array([[p.x, p.y] for p in predictor(input_img, rects[0]).parts()])
    landmark[:, 1] = input_img.shape[0] - 1 - landmark[:, 1]
    draw_landmarks(input_img, landmark, os.path.join(save_path, img_name), [0, 0, 255])
    np.savetxt(os.path.join(save_path, img_name.split('.')[0]+'.txt'), landmark)
    return os.path.join(save_path, img_name), os.path.join(save_path, img_name.split('.')[0]+'.txt')

def get_3DPoints(save_path, img_name, pred_vertex):
    landmarks = pred_vertex.cpu().numpy()
    lm_path = os.path.join(save_path, img_name.split('.')[0] + '.txt')
    with open(lm_path, "w") as f:
        for lm in landmarks[0]:
            f.write(str(lm[0]) + '\t' + str(lm[1]) + '\t' + str(lm[2]))
            f.write('\n')
    f.close()
    return lm_path

def get_3D68_landmarks(keypoints, save_path, img_name, pred_vertex):
    landmarks = pred_vertex[:, keypoints].cpu().numpy()
    lm_path = os.path.join(save_path, img_name.split('.')[0] + '.txt')
    with open(lm_path, "w") as f:
        for lm in landmarks[0]:
            f.write(str(lm[0]) + '\t' + str(lm[1]) + '\t' + str(lm[2]))
            f.write('\n')
    f.close()
    return lm_path

def get_pose(save_path, img_name, pose):
    angle, trans = pose['angle'].cpu().numpy().squeeze(), pose['trans'].cpu().numpy().squeeze()
    pose_path = os.path.join(save_path, img_name.split('.')[0] + '.txt')
    with open(pose_path, "w") as f:
        f.write(str(angle[0]) + '\t' + str(angle[1]) + '\t' + str(angle[2]))
        f.write('\n')
        f.write(str(trans[0]) + '\t' + str(trans[1]) + '\t' + str(trans[2]))
    f.close()

def perspective_projection(focal, center):
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

def back_perspective_projection(focal, center):
    return np.array([
        1/focal, 0, -center/focal,
        0, 1/focal, -center/focal,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()

def D2toD3(save_path, path2d, path3d):
    lm = np.loadtxt(path2d).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm = np.hstack((lm, np.ones([lm.shape[0],1])))

    back_persc_proj = back_perspective_projection(focal, center)
    lm_ = lm @ back_persc_proj

    lm3d = np.loadtxt(path3d).astype(np.float32)
    lm3d = lm3d.reshape([-1, 3])
    lm3d_Z = lm3d[...,2:]

    lm_ = lm_*lm3d_Z

    np.savetxt(os.path.join(save_path, path2d.split('/')[-1]), lm_)

    return os.path.join(save_path, path2d.split('/')[-1])

def D3toD2(save_path, im, path):
    im = Image.open(im).convert('RGB')
    im = np.array(im)[:,:,::-1]

    lm = np.loadtxt(path).astype(np.float32)
    lm = lm.reshape([-1, 3])
    persc_proj = perspective_projection(focal, center)
    lm_ = lm @ persc_proj
    lm_ = lm_[..., :2] / lm_[..., 2:]

    draw_landmarks(im, lm_, os.path.join(save_path, path.split('/')[-1].replace('.txt','.jpg')), [255, 0, 0])
    np.savetxt(os.path.join(save_path, path.split('/')[-1]), lm_)

    return os.path.join(save_path, path.split('/')[-1])

def main(rank, opt, name='examples'):
    p1,p2,p3,p4,p5,p6 = makedirs(name)

    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)  # models.facerecon_model.FaceReconModel
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder)  # [5,3]

    for i in range(len(im_path)):
        print(i, im_path[i])
        start = time.time()
        # img_name = im_path[i].split(os.path.sep)[-1].replace('.png', '').replace('.jpg', '')
        img_name = im_path[i].split('/')[-1]
        if not os.path.isfile(lm_path[i]):
            continue
        im, lm, im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'test_vid': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)  # unpack data from data loader
        end = time.time()
        print('Img Process complete in {:.3f}ms'.format((end - start) * 1000))

        model.test()  # run inference

        # 2D 68 landmarks
        # lm_sess, input_op, output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb')
        # lm2D_img_path,lm2D_txt_path = get_2D68_landmarks(name, img_name, im, lm_sess, input_op, output_op)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./checkpoints/lm_model/shape_predictor_68_face_landmarks.dat')
        start = time.time()
        lm2D_img_path, lm2D_txt_path = get_2D68_landmarks_dlib(p1, img_name, im, detector, predictor)
        end=time.time()
        print('Get 2D 68 landmarks complete in {:.3f}ms'.format((end - start) * 1000))

        # 3D points
        pred_vertex = model.pred_vertex
        start = time.time()
        points3D_path = get_3DPoints(p2, img_name, pred_vertex)
        end = time.time()
        print('Get 3D Points complete in {:.3f}ms'.format((end - start) * 1000))

        # 3D 68 landmarks
        bfm = loadmat('./BFM/BFM_model_front.mat')
        keypoints = np.squeeze(bfm['keypoints']).astype(np.int64) - 1
        start = time.time()
        lm3D_path = get_3D68_landmarks(keypoints, p3, img_name, pred_vertex)
        end = time.time()
        print('Get 3D 68 landmarks complete in {:.3f}ms'.format((end - start) * 1000))

        # pose
        pose = model.pred_coeffs_dict
        start = time.time()
        pose_path = get_pose(p4, img_name, pose)
        end = time.time()
        print('get pose complete in {:.3f}ms'.format((end - start) * 1000))

        #2dto3d
        start = time.time()
        lm2Dto3D_txt_path = D2toD3(p5, lm2D_txt_path, lm3D_path)
        end = time.time()
        print('2Dto3D complete in {:.3f}ms'.format((end - start) * 1000))

        #3dto2d
        start = time.time()
        lm3Dto2D_txt_path = D3toD2(p6, lm2D_img_path, lm3D_path)
        end = time.time()
        print('3Dto2D complete in {:.3f}ms'.format((end - start) * 1000))

if __name__ == '__main__':
    opt = TestOptions().parse()
    # get_5_landmarks(opt.img_folder)
    main(0, opt, opt.img_folder)