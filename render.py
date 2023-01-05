import argparse
import numpy as np
import cv2
import torch
from scipy.io import loadmat, savemat
from models.bfm import ParametricFaceModel
from util.nvdiffrast import MeshRenderer
import trimesh
import os

def save_mesh(vertex, color, face_buf, name):
    recon_shape = vertex  # get reconstructed shape
    recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
    recon_shape = recon_shape.cpu().numpy()[0]
    recon_color = color
    recon_color = recon_color.cpu().numpy()[0]
    tri = face_buf.cpu().numpy()
    mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                           vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
    mesh.export(name)

def render(args):
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)

    facemodel = ParametricFaceModel(
        bfm_folder=args.bfm_folder, camera_distance=args.camera_d, focal=args.focal,
        center=args.center, is_train=False, default_name=args.bfm_model
    )
    device = torch.device(0)
    facemodel.to(device)

    fov = 2 * np.arctan(args.center / args.focal) * 180 / np.pi
    renderer = MeshRenderer(
        rasterize_fov=fov, znear=args.z_near, zfar=args.z_far, rasterize_size=int(2 * args.center)
    )

    # 表情替换
    # idmat = loadmat(args.idmatFile)
    # exmat = loadmat(args.exmatFile)

    # coeffs = np.zeros(shape=(1, 257))
    # coeffs[:, :80]=idmat['id']
    # coeffs[:, 80: 144]=exmat['exp']
    # coeffs[:, 144: 224]=idmat['tex']
    # coeffs[:, 224: 227]=idmat['angle']
    # coeffs[:, 227: 254]=idmat['gamma']
    # coeffs[:, 254:]=idmat['trans']

    # coeffs = torch.from_numpy(np.float32(coeffs))
    # coeffs = coeffs.to(device)

    # vertex, tex, color, lm = facemodel.compute_for_render(coeffs)
    # mask, _, face = renderer(vertex, facemodel.face_buf, feat=color)

    # id_img = cv2.imread(args.idmatFile.replace('.mat', '.png'))
    # id_img = id_img[:, :, ::-1]
    # id_img = id_img[:, :448, :]

    # ex_img = cv2.imread(args.exmatFile.replace('.mat', '.png'))
    # ex_img = ex_img[:, :, ::-1]
    # ex_img = ex_img[:, :448, :]
    # ex_img = torch.from_numpy(np.float32(ex_img / 255)).permute(2, 0, 1).unsqueeze(0)
    # ex_img = ex_img.to(device)

    # output_vis = face * mask
    # output_vis = face* mask + (1 - mask) * ex_img
    # output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

    # output_vis_numpy = np.concatenate((id_img,ex_img,output_vis_numpy_raw[0]), axis=-2)
    # output_vis_numpy = output_vis_numpy[:, :, ::-1]

    # name = args.idmatFile.split('/')[-1].replace('.mat','')+'_'+args.exmatFile.split('/')[-1].replace('.mat','')
    # cv2.imwrite(args.savePath + name + '.jpg', output_vis_numpy_raw[0][:, :, ::-1])

    # 64个表情基
    for i in range(80,144):
        coeffs = np.zeros(shape=(1, 257))
        coeffs[:, i] = 1
        coeffs = torch.from_numpy(np.float32(coeffs))
        coeffs = coeffs.to(device)

        vertex, tex, color, lm = facemodel.compute_for_render(coeffs)
        mask, _, face = renderer(vertex, facemodel.face_buf, feat=color)
        coeffs_dict = facemodel.split_coeff(coeffs)

        output_vis = face * mask
        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

        name = str(i - 79)
        cv2.imwrite(args.savePath+name+'.jpg',output_vis_numpy_raw[0][:, :, ::-1])

        save_mesh(vertex,color,facemodel.face_buf,args.savePath+name+'.obj')

def concatenate_exp():
    paths=['exp-2','exp-1','exp1']
    savepath = os.path.join('./render/', 'exp-2-1+1')

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for i in range(1, 65):
        name=str(i)+'.jpg'
        imgs=[]
        for path in paths:
            imgs.append(os.path.join('./render/',path,name))
        img_con=np.concatenate((cv2.imread(imgs[0]),cv2.imread(imgs[1]), cv2.imread(imgs[2])), axis=-2)
        cv2.imwrite(os.path.join(savepath, name), img_con)

def get_exp(path,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for file in sorted(os.listdir(path)):
        if file.endswith('mat'):
            f = open(os.path.join(savepath, file.split('.')[0]+'.txt'), "w")
            mat=loadmat(os.path.join(path, file))
            exp=mat['exp']

            for i in range(len(exp[0])):
                f.write(str(exp[0][i])+'\n')
            f.close()

def test_exp(args,idmatFile=None):
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)

    facemodel = ParametricFaceModel(
        bfm_folder=args.bfm_folder, camera_distance=args.camera_d, focal=args.focal,
        center=args.center, is_train=False, default_name=args.bfm_model
    )

    device = torch.device(0)
    facemodel.to(device)

    fov = 2 * np.arctan(args.center / args.focal) * 180 / np.pi
    renderer = MeshRenderer(
        rasterize_fov=fov, znear=args.z_near, zfar=args.z_far, rasterize_size=int(2 * args.center)
    )
    if idmatFile!=None:
        idmat = loadmat(idmatFile)

    for file in sorted(os.listdir(args.exmatPath)):
        if file.endswith('mat'):
            exmat = loadmat(os.path.join(args.exmatPath,file))

            coeffs = np.zeros(shape=(1, 257))
            coeffs[:, 80: 144]=exmat['exp']

            if idmatFile!=None:
                coeffs[:, :80] = idmat['id']
                coeffs[:, 144: 224]=idmat['tex']
                coeffs[:, 224: 227]=idmat['angle']
                coeffs[:, 227: 254]=idmat['gamma']
                coeffs[:, 254:]=idmat['trans']

            coeffs = torch.from_numpy(np.float32(coeffs))
            coeffs = coeffs.to(device)

            vertex, tex, color, lm = facemodel.compute_for_render(coeffs)
            mask, _, face = renderer(vertex, facemodel.face_buf, feat=color)

            output_vis = face * mask
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

            name = str(file.split('.')[0])
            cv2.imwrite(args.savePath + name + '.jpg', output_vis_numpy_raw[0][:, :, ::-1])

            save_mesh(vertex, color, facemodel.face_buf, args.savePath + name + '.obj')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch Deep3DFace render')
    parser.add_argument('--bfm_folder', type=str, default='BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    # parser.add_argument('--idmatFile', type=str, default='./checkpoints/pretraining_model/results/qss_vid/epoch_20_000000/001.mat')
    # parser.add_argument('--exmatFile', type=str, default='./checkpoints/pretraining_model/results/ly_vid/epoch_20_000000/228.mat')
    parser.add_argument('--exmatPath', type=str, default='./render/get_classical_exp')
    parser.add_argument('--savePath', type=str,default='./render/classical_drive_exp/')
    args = parser.parse_args()

    # render(args)
    # concatenate_exp()
    # get_exp('./render/get_classical_exp','./render/get_classical_exp')
    test_exp(args,'./checkpoints/pretraining_model/results/qss_vid/epoch_20_000000/001.mat')