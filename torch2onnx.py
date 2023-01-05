import torch
from models.facerecon_model import FaceReconModel
from options.test_options import TestOptions

opt = TestOptions().parse()
dummy_input = torch.randn(1, 3, 224, 224)

model = FaceReconModel(opt)

checkpoint = './checkpoints/pretraining_model/epoch_20.pth'
onnx_path = './checkpoints/pretraining_model/epoch_20.onnx'
state_dict = torch.load(checkpoint)
net = model.net_recon
net.load_state_dict(state_dict['net_recon'])

torch.onnx.export(net, dummy_input, onnx_path,verbose=True)
print("Exporting .pth model to onnx model has been successful!")


