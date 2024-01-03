import argparse
import os
from collections import OrderedDict
import math
import time

import numpy as np
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import collections

from data.dataset_3d import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs/reproduce_pointbert_8kpts', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='ROCA', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='mytest', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    # Training
    parser.add_argument('--epochs', default=2050, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=13, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=20, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='outputs/reproduce_pointbert_8kpts/checkpoint_2001.pt', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    # parser.add_argument('--test_ckpt_addr', default='ckpts/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt', help='the ckpt to test 3d zero shot')
    # parser.add_argument('--test_ckpt_addr', default='outputs/reproduce_pointbert_8kpts/checkpoint_best.pt', help='the ckpt to test 3d zero shot')
    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    return parser

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def main(args):

    # image load
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        normalize
    ])
    image_path = 'data/ROCA/rendered_images/02747177-75dbeb01fdc7f265d0a96520c31993ad/02747177-75dbeb01fdc7f265d0a96520c31993ad_0000.png'
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = train_transform(img).unsqueeze(dim=0).cuda()

    # text/label load
    taxonomy_file = 'data/ROCA/taxonomy.json'
    with open(taxonomy_file,'rb') as f:
        taxonomies = json.load(f)
    catid_cad = image_path.split('/')[-1].split('-')[0]
    thing_classes = []
    tokenized_captions = []
    for tax in taxonomies:
        if tax['synsetId']==catid_cad:
            thing_classes = [cat for cat in tax['name'].split(',')]
    caption = thing_classes[0]
    tokenizer = SimpleTokenizer()
    tokenized_captions.append(tokenizer(caption))
    tokenized_captions = torch.stack(tokenized_captions).cuda()

    # load point cloud
    pc_dir = 'data/ROCA/roca_pc'
    filename_list = os.listdir(pc_dir)
    pc_list = []
    for filename in filename_list:
        pc = np.load(os.path.join(pc_dir,filename)).astype(np.float32)
        pc = pc_norm(pc)
        pc_list.append(torch.from_numpy(pc).cuda())
    # pc = pc_list[0].unsqueeze(dim=0)

    # model setting
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))

    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    model.eval()

    logit_scale = torch.tensor(76.0,device='cuda')
    with torch.no_grad():
        # text encoder
        text_embed_all = []

        text_for_one_sample = tokenized_captions
        text_embed = model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed_all.append(text_embed)

        text_embed = torch.stack(text_embed_all)

        # pc_encoder
        pc_embed_all = []
        for pc in pc_list:
            pc_embed = model.encode_pc(pc.unsqueeze(0))
            pc_embed_all.append(pc_embed.squeeze())
            print("------------------> processing pc !!! --------------------------> ")
        pc_embed_all = torch.stack(pc_embed_all)
        with open('ULIP_PointBERT_cads_feature.pkl','wb') as f:
            pickle.dump((pc_embed_all,filename_list),f)

        # image encoder
        image_embed = model.encode_image(img)

        # normalized features
        pc_embed_all = F.normalize(pc_embed_all, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # cosine similarity as logits
        # logits_per_pc_text = logit_scale * pc_embed_all @ text_embed.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        # logits_per_pc_image = logit_scale * pc_embed_all @ image_embed.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        image_retrival = logits_per_image_pc.argmax()
        text_retrival = logits_per_text_pc.argmax()
        print(f'Image retrival result : {filename_list[image_retrival].split(".")[0]}')
        print(f'Text retrival result : {filename_list[text_retrival].split(".")[0]}')

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        plt.figure()
        plt.imshow(image)
        plt.title(caption)
        plt.show()

        pc = np.load(os.path.join(pc_dir,filename_list[image_retrival])).astype(np.float32)
        image_pc = o3d.geometry.PointCloud()
        image_pc.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([image_pc])

        pc = np.load(os.path.join(pc_dir,filename_list[text_retrival])).astype(np.float32)
        text_pc = o3d.geometry.PointCloud()
        text_pc.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([text_pc])


    print('hello world !')


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('ULIP demo', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    duration = end_time - start_time
    print("程序运行时间：", duration, "秒")