"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_data_loader_face,get_test_loader_face, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, plot_batch
import argparse
from torch.autograd import Variable
from trainer import FACE_Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='FACE', help="MUNIT|FACE|CITY")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
crop_image_height = config['crop_image_height']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'FACE':
    trainer = FACE_Trainer(config)                                  # 在定义网络的类的__init__方法中，创建模型
else:
    sys.exit("Only support MUNIT|CITY|FACE")
trainer.cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

if opts.trainer == 'FACE':
	# Start training
    train_loader = get_data_loader_face(config)
    test_loader = get_test_loader_face(config)
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:                                                         # 控制总的steps
    	for it, (images_a, images_b) in enumerate(train_loader):        # 在每一个epoch中迭代step
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            # with Timer("Elapsed time in update: %f"):
            	# Main training code
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['print_log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            if (iterations + 1) % config['log_iter'] == 0:
                # print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:           # 每次显示8张图片
                test_image_a, test_image_b = next(iter(test_loader))        # 返回8个图像、8个关键点热图，
                                                                            # 为什么每次都是那 8 个人
                test_image_outputs = []
                test_image_outputs.append(np.zeros((3,crop_image_height,crop_image_height)))    # 图像中左上角没有图像的灰色区域
                for i in range(display_size):
                    boundary = test_image_b[i].sum(0).unsqueeze(0)
                    test_image_outputs.append(np.tile(boundary.numpy(),(3,1,1)))
                for i in range(display_size):
                    test_image_outputs.append(test_image_a[i,...].numpy())
                    for j in range(display_size):
                        with torch.no_grad():
                            out = trainer.transfer(test_image_a[i].unsqueeze(0).cuda(),test_image_b[j].unsqueeze(0).cuda())[-1]   # 本质上就是调用训练的网络参数，输入图像、要迁移的风格的关键点热图
                                                                                                                                  # 给出风格迁移之后的图像
                        test_image_outputs.append(out[0,...].data.cpu().numpy())
                # for i in range(len(test_image_outputs)):
                #     print(test_image_outputs[i].shape)
                test_image_outputs = np.stack(test_image_outputs, axis = 0)             # 此时维度为：(81, 3, width, height)
                test_image_outputs = test_image_outputs.transpose((0,2,3,1))            # 将(b, c, w, h)转换为(b, w, h, c)的形式
                plot_batch(test_image_outputs, os.path.join(image_directory,"test_{:08}.png".format(iterations + 1)))

            # Save network weights                                      # 保存模型，并判断是否终止训练
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)          # 仅存储model的state_dict参数，以{"gen_weight": state_dict()}的形式
            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
