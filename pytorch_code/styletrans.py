import os
from networks import NetV2_128x128
from trainer import FACE_Trainer
import torch
from new_dataloader import load_img, points_to_landmark_map, FaceDataset
from utils import get_test_loader_face, get_config, postprocess, plot_batch
from PIL import Image
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboard
from networks import NetV2_128x128
import torch.onnx as onnx
import cv2
from tqdm import tqdm


def stylestrans_single():
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

def stylestrans_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test_0002.yaml',
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
    test_loader = get_test_loader_face(config)                          # 加载图像
    land_offer = get_test_loader_face(config)

    display_size = config['display_size']
    crop_image_height = config['crop_image_height']
    image_directory = config['root_dir']

    with torch.no_grad():
        for i, (img, _) in tqdm(enumerate(test_loader)):                  # 连续取出10张图片， 设置batch_size == 10
            for _, landmap in land_offer:                               # 10个关键点map
                # results = []
                # # landmap = landmap
                #
                # for k in range(10):                                     # 按列进行处理，每一个关键点对应一列
                #     land = landmap[k]
                #     land_size = land.shape
                #     land = land.repeat(10).reshape((10, *land_size))
                #     img1_stylized = net(img, land)[-1]
                #
                #     img1_stylized = img1_stylized.data.cpu().numpy().transpose((0, 2, 3, 1))
                #     img1_stylized = postprocess(img1_stylized)
                #     results.append(img1_stylized)

                # test_image_a, test_image_b = next(iter(test_loader))  # 返回8个图像、8个关键点热图，

                test_image_a = img
                test_image_b = landmap


                test_image_outputs = []
                test_image_outputs.append(np.zeros((3,crop_image_height, crop_image_height)))           # 图像中左上角没有图像的灰色区域
                for j in range(display_size):
                    boundary = test_image_b[j].sum(0).unsqueeze(0)
                    test_image_outputs.append(np.tile(boundary.numpy(),(3,1,1)))
                for j in range(display_size):
                    test_image_outputs.append(test_image_a[j,...].numpy())
                    for j in range(display_size):
                        with torch.no_grad():                   # 调用训练的网络参数，输入一张图像、一个关键点热图，进行风格迁移， 给出风格迁移之后的图像
                            out = net(test_image_a[j].unsqueeze(0).cuda(), test_image_b[j].unsqueeze(0).cuda())[-1]

                        test_image_outputs.append(out[0,...].data.cpu().numpy())
                test_image_outputs = np.stack(test_image_outputs, axis = 0)                              # 此时维度为：(81, 3, width, height)
                test_image_outputs = test_image_outputs.transpose((0,2,3,1))                             # 将(b, c, w, h)转换为(b, w, h, c)的形式
                plot_batch(test_image_outputs, os.path.join(image_directory, "test_{:08}.png".format(i + 1)))

        # # 显示两张原始图片，以及风格迁移后的图片
        # img1 = img1[0].data.cpu().numpy().transpose((1, 2, 0))
        # img2 = img2[0].data.cpu().numpy().transpose((1, 2, 0))
        # img1 = postprocess(img1)  # 将X的值映射到 0-255的范围内，同时转换成正整数
        # img2 = postprocess(img2)  # 将X的值映射到 0-255的范围内，同时转换成正整数
        #
        # img1 = np.array([img1[..., 2], img1[..., 1], img1[..., 0]]).transpose((1, 2, 0))
        # img2 = np.array([img2[..., 2], img2[..., 1], img2[..., 0]]).transpose((1, 2, 0))
        # img1_stylized = np.array([img1_stylized[..., 2], img1_stylized[..., 1], img1_stylized[..., 0]]).transpose((1, 2, 0))
        # print(img1.shape)
        # cv2.imshow("图1", img1)
        # cv2.imshow("图2", img2)
        # cv2.imshow("合成图，图1的风格，图2的关键点", img1_stylized)
        # cv2.waitKey(0)




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

    # stylestrans_single()
    stylestrans_batch()


