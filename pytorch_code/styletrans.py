import os
from networks import NetV2_128x128
from trainer import FACE_Trainer
import torch
from new_dataloader import load_img, points_to_landmark_map, FaceDataset
from utils import get_test_loader_face, get_config, postprocess
from PIL import Image
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboard
from networks import NetV2_128x128
import torch.onnx as onnx
import cv2


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml',
                        help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='FACE', help="MUNIT|FACE|CITY")
    opts = parser.parse_args()

    # read pt file, and get "gen_weight"
    model_path = "./outputs/exp_0001/checkpoints/gen_01000000.pt"
    with open(model_path, 'rb') as mp:
        model = torch.load(mp)
    model_state = model["gen_weight"]

    # init model and load_state_dict
    net = NetV2_128x128(3, 3, 3)
    net.load_state_dict(model_state)
    net.cuda()

    # load src image, landmark and style image
    config = get_config(opts.config)
    test_loader = get_test_loader_face(config)              # 加载图像
    test_loader = iter(test_loader)
    img1, struct_map1 = next(test_loader)  # 是RGB、float32、(b, c, h, w), torch
    img2, struct_map2 = next(test_loader)

    img1 = img1.cuda().detach()
    struct_map2 = struct_map2.cuda().detach()
    img1_stylized = net(img1, struct_map2)[-1]
    img1_stylized = img1_stylized[0].data.cpu().numpy().transpose((1, 2, 0))
    img1_stylized = postprocess(img1_stylized)
    Image.fromarray(img1_stylized).save('/home/shuai.li/dset/WFLW/temp/temp1.jpg')

    # 显示两张原始图片，以及风格迁移后的图片
    img1= img1[0].data.cpu().numpy().transpose((1, 2, 0))
    img2= img2[0].data.cpu().numpy().transpose((1, 2, 0))
    img1 = postprocess(img1)              # 将X的值映射到 0-255的范围内，同时转换成正整数
    img2 = postprocess(img2)              # 将X的值映射到 0-255的范围内，同时转换成正整数

    img1 = np.array([img1[..., 2], img1[..., 1], img1[..., 0]]).transpose((1,2,0))
    img2 = np.array([img2[..., 2], img2[..., 1], img2[..., 0]]).transpose((1,2,0))
    img1_stylized = np.array([img1_stylized[..., 2], img1_stylized[..., 1], img1_stylized[..., 0]]).transpose((1,2,0))
    print(img1.shape)
    cv2.imshow("图1", img1)
    cv2.imshow("图2", img2)
    cv2.imshow("合成图，图1的风格，图2的关键点", img1_stylized)
    cv2.waitKey(0)




if __name__ == "__main__":

    # import torch.nn as nn
    # class netdef(nn.Module):
    #     def __init__(self):
    #         super(netdef, self).__init__()
    #         self.conv = nn.Conv2d(3, 3, 3)
    #     def forward(self, input):
    #         data = self.conv(input)
    #         return data
    #
    # writer = SummaryWriter(logdir='./netinfo/graph')
    # net = NetV2_128x128(3, 3, 3)
    # # net = netdef()
    # input = torch.arange(128*128*3*2, dtype=torch.float).reshape((2, 3, 128, 128))
    # # onnx.export(net, input, './onnx')
    #
    # # writer.add_graph(net, input_to_model = (input, ))
    # # writer.close()
    # hm = torch.arange(128 * 128 * 3 * 2, dtype=torch.float).reshape((2, 3, 128, 128))
    # writer.add_graph(net, input_to_model = (input, hm))
    # writer.close()
    # onnx.export(net, (input, hm), './onnx')

    test()







