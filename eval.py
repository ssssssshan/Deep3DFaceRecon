import argparse
from models import networks
from deepface import DeepFace
import torch.nn.functional as F
from models.arcface_torch.backbones import get_model
from util import preprocess
import numpy as np
from PIL import Image
import cv2
import os
import dlib
import torch
import time
import matplotlib.pyplot as plt
from scipy.integrate import simps
import math
from urllib import request
import requests
import json
import base64
import shutil

def l2_distance(input1: np.ndarray, input2: np.ndarray) -> float:
    return np.linalg.norm(input1-input2)

def l1_distance(input1: np.ndarray, input2: np.ndarray) -> float:
    return np.linalg.norm(input1-input2, ord=1)

def IP_distance(input1: np.ndarray, input2: np.ndarray) -> float:
    return np.dot(input1, input2)/np.linalg.norm(input1)/np.linalg.norm(input2)

def get_data_path(root):
    input_path = [line.replace("\n","") for line in open(root+"input.txt","r").readlines()]
    res_path = [line.replace("\n","") for line in open(root+"res.txt","r").readlines()]
    return input_path, res_path

def sigmoid(z,a,b):
    fz = []
    for num in z:
        fz.append(1 / (1 + math.exp(a*num+b)))
    return fz

def AUCError(errors, failureThreshold=8, step=0.01, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x / 100])) / nErrors for x in xAxis]
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

def nme(pred_pts, gt_pts):
    base = np.linalg.norm(gt_pts[36] - gt_pts[45])  # outter corners of eyes
    nme = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1) / base)
    return nme

def get_lm(detector, predictor, img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 1)

    if len(rects) != 1:
        print('*******detection {} failed********'.format(img_path))
        return []
    else:
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[0]).parts()])
        return landmarks

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img

def read_data(img_path, to_tensor=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./checkpoints/lm_model/shape_predictor_68_face_landmarks.dat')

    lm = get_lm(detector,predictor,img_path)

    if len(lm)==0:
        trans_m=None
    else:
        lm = torch.tensor(lm).unsqueeze(0)
        trans_m = preprocess.estimate_norm_torch(lm, img.shape[-2])

    if to_tensor:
        img=torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img, trans_m

def make_eval(img_path, save_dir):
    f1 = open(save_dir + "input.txt", "w")
    f2 = open(save_dir + "res.txt", "w")
    i = 0
    for img in sorted(os.listdir(img_path)):
        if img.endswith('png') or img.endswith('jpg'):
            file = os.path.join(img_path, img)
            image = cv2.imread(file)

            img1_path = save_dir + "input/" + str(i).zfill(4) + "." + img.split(".")[-1]
            img2_path = save_dir + "res/" + str(i).zfill(4) + "." + img.split(".")[-1]
            cv2.imwrite(img1_path, image[:, :1024, :])
            cv2.imwrite(img2_path, image[:, 1024:2048, :])

            f1.write(img1_path+'\n')
            f2.write(img2_path+'\n')

            if i%100==0:
                print(i)
            i+=1
    print("save to", save_dir + "input.txt", "and", save_dir + "res.txt")
    f1.close()
    f2.close()

def make_raw(data_path,save_dir):
    if data_path=='30000/':
        img_path='./datasets/FFQH/images1024x1024/30000/'
    elif data_path=='test/':
        img_path='./datasets/faces_glintasia/test/'
    elif data_path=='test2/':
        img_path = './datasets/test2/'

    f = open(save_dir + "raw.txt", "w")
    i = 0
    for file in sorted(os.listdir(img_path)):
        if file.endswith('png') or file.endswith('jpg'):
            if not os.path.isfile(img_path+"detections/"+fileç.replace('.jpg','.txt')):
                continue
            file_path = os.path.join(img_path, file)
            save_path = save_dir + "raw/" + str(i).zfill(4) + "." + file.split(".")[-1]
            shutil.copyfile(file_path, save_path)
            f.write(save_path + '\n')

            if i % 100 == 0:
                print(i)
            i += 1
    print("save to", save_dir + "raw.txt")
    f.close()

def get_dis_trans(save_dir):
    net_recog = networks.define_net_recog(net_recog="r100",
                                          pretrained_path="checkpoints/recog_model/ms1mv3_arcface_r100_fp16/backbone.pth")
    distances = []
    input_path, res_path = get_data_path(save_dir)
    for i in range(len(input_path)):
        img1, trans_m1 = read_data(input_path[i])
        img2, trans_m2 = read_data(res_path[i])

        if(trans_m1==None or trans_m2==None):
            continue

        feat1 = net_recog(img1, trans_m1).detach().cpu().numpy()
        feat2 = net_recog(img2, trans_m2).detach().cpu().numpy()
        distance = IP_distance(feat1[0], feat2[0])
        distances.append(distance)
        print(i, input_path[i], distance)
    return distances

def get_cos_dis(save_dir):
    net = get_model("r100", fp16=True)
    net.load_state_dict(torch.load("checkpoints/recog_model/ms1mv3_arcface_r100_fp16/backbone.pth"))
    net.eval()

    distances = []
    # thresholds = np.arange(0, 4, 0.001)
    input_path, res_path = get_data_path(save_dir)
    for i in range(len(input_path)):
        img1 = read_img(input_path[i])
        img2 = read_img(res_path[i])

        feat1 =net(img1).detach().numpy()
        feat2 = net(img2).detach().numpy()

        distance = IP_distance(feat1[0], feat2[0])
        distances.append(distance)
        print(i, input_path[i], distance)

    # a=(np.log(0.0526)-np.log(9))/0.3
    # b=np.log(9)-0.1*a
    # distances=sigmoid(distances,a,b)
    return distances

def get_dis_deepface(save_dir):
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", ]
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    model = DeepFace.build_model(models[6])

    distances = []
    input_path, res_path = get_data_path(save_dir)
    for i in range(len(input_path)):
        res = DeepFace.verify(img1_path=input_path[i],img2_path=res_path[i], model_name=models[6], distance_metric=metrics[0], detector_backend=backends[3], enforce_detection=True)
        print(res)

        # feat1 = DeepFace.represent(img_path=input_path[i], model_name=models[6], model=model, detector_backend=backends[3], enforce_detection=False)
        # feat2 = DeepFace.represent(img_path=res_path[i], model_name=models[6], model=model, detector_backend=backends[3], enforce_detection=False)
        # distance = IP_distance(feat1, feat2)
        # distances.append(distance)
        # print(i, input_path[i], distance)
    return distances

def gettoken():
    ak = '4BofeyM2qxbYOVVtAGVntw69' #获取到的API Key
    sk = 'x2OAUjmuIu2xH6Gw5OIAyVMpIVz91Hq5' #获取到的Secret Key
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+ak+'&client_secret='+sk
    response = requests.get(host)
    if response:
        print('get token success', response.json())
    return response.json()['access_token']

def to_base64(file_name):
    with open(file_name,'rb') as f:
        base64_data = base64.b64encode(f.read())
        image = str(base64_data,'utf-8')
    return image

def get_api(save_dir):
    request_url="https://aip.baidubce.com/rest/2.0/face/v3/match?access_token=" + gettoken()
    scores = []
    input_path, res_path = get_data_path(save_dir)
    for i in range(len(input_path)):
        image1=to_base64(input_path[i])
        image2=to_base64(res_path[i])
        params = json.dumps(
            [{"image": image1, "image_type": "BASE64", "face_type": "LIVE", "quality_control": "LOW"},
             {"image": image2, "image_type": "BASE64", "face_type": "LIVE", "quality_control": "LOW"}]).encode(encoding='UTF8')
        my_request = request.Request(url=request_url, data=params)
        time.sleep(0.5)
        my_request.add_header('Content-Type', 'json')
        # urlencode处理需提交的数据
        null=0
        response = request.urlopen(my_request,timeout=2)
        result = eval(bytes.decode(response.read()))
        if result['error_msg'] == 'SUCCESS':
            score = result['result']['score']
            print(i, input_path[i], '两张图片相似度：', score)
            scores.append(score)
        else:
            print('错误信息：', result['error_msg'])
    return scores

def get_nme(save_dir,showcurve):
    PREDICTOR_PATH = "./checkpoints/lm_model/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    nmes = []
    input_path, res_path = get_data_path(save_dir)
    for i in range(len(input_path)):
        lm1 = get_lm(detector, predictor, input_path[i])
        lm2 = get_lm(detector, predictor, res_path[i])

        if len(lm1)==0 or len(lm2)==0:
            continue

        res = nme(lm2, lm1)
        nmes.append(res)
        print(i, input_path[i], res)

    print("IMG_PATH: {}".format(save_dir))
    print("NME:{}".format(np.mean(nmes)))

    AUCError(nmes,showCurve=showcurve)

def get_cos_single(model_path,data_path):
    save_dir = "./checkpoints/compare/eval/" + model_path + data_path

    net = get_model("r100", fp16=True)
    net.load_state_dict(torch.load("checkpoints/recog_model/ms1mv3_arcface_r100_fp16/backbone.pth"))
    net.eval()

    # input_path=save_dir+"input/0019.png"
    # res_path=save_dir+"res/0019.png"
    input_path="./datasets/bjt/3.jpg"
    res_path = "./datasets/bjt/4.jpg"

    img1 = read_img(input_path)
    img2 = read_img(res_path)

    feat1 = net(img1).detach().numpy()
    feat2 = net(img2).detach().numpy()

    distance = IP_distance(feat1[0], feat2[0])
    print(input_path, distance)

def main(model_path,data_path):
    # img_path = "./checkpoints/" + model_path + "results/" + data_path + "epoch_20_000000"
    img_path = "./checkpoints/" + model_path + "results/" + data_path
    save_dir = "./checkpoints/compare/eval/" + model_path + data_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+"input")
        os.makedirs(save_dir + "res")
        print("mkdir eval")
        make_eval(img_path,save_dir)

    # os.makedirs(save_dir + "raw")
    # make_raw(data_path, save_dir)

    # distances1 = get_dis_trans(save_dir)
    # print("cosine：", np.mean(distances1))

    distances2 = get_cos_dis(save_dir)
    print("cosine：", np.mean(distances2))

    # distances3 = get_dis_deepface(save_dir)

    scores = get_api(save_dir)
    print("scores：", np.mean(scores))

    get_nme(save_dir,True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch Deep3DFace eval')
    parser.add_argument('--model_path', type=str, default='pretraining_model/', help='backbone network')
    parser.add_argument('--data_path', type=str, default='3s/')
    args = parser.parse_args()

    start = time.time()
    main(args.model_path, args.data_path)
    # get_cos_single(args.model_path, args.data_path)
    end = time.time()
    print('complete in {:.0f}min {:.2f}s'.format((end - start)//60,(end - start)%60))