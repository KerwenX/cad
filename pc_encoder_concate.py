import numpy as np
import os
import torch
from utils import utils
from collections import OrderedDict
import warnings
import logging
import models.ULIP_models as models
import argparse
import pickle
import time
# log_format = '%(asctime)s - DEMO4 - %(levelname)s - %(message)s'
# logging.basicConfig(filename='demo_2.out.log', level=logging.INFO, filemode='w',format=log_format)
# logging.getLogger().addHandler(logging.StreamHandler())
warnings.filterwarnings("ignore", category=UserWarning)

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP evaluation', add_help=False)
    # Data
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    # System
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    # parser.add_argument('--test_ckpt_addr', default='ckpts/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt', help='the ckpt to test 3d zero shot')
    # parser.add_argument('--test_ckpt_addr', default='outputs/reproduce_pointbert_8kpts/checkpoint_best.pt', help='the ckpt to test 3d zero shot')
    parser.add_argument('--test_ckpt_addr', default='/home/aston/Desktop/python/CAD-Matching/ckpts/concate/checkpoint_1.pt', help='the ckpt to test 3d zero shot')
    # parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    return parser

start_time = time.time()
parser = argparse.ArgumentParser('ULIP demo', parents=[get_args_parser()])
args = parser.parse_args()
def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

pc_file = 'data/ROCA/roca_pc'
filename_list = os.listdir(pc_file)

seed = 2023 + utils.get_rank()
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

logit_scale = torch.tensor(76.0, device='cuda')

# logging.getLogger().addHandler(logging.StreamHandler())
with torch.no_grad():
    pc_embed_all_2 = []
    for filename in filename_list:
        pc = np.load(os.path.join(pc_file, filename)).astype(np.float32)
        pc = pc_norm(pc)
        pc = torch.from_numpy(pc).cuda()
        pc_embed = model.encode_pc(pc.unsqueeze(0))
        pc_embed_all_2.append(pc_embed.squeeze())
        # print("--------->{} pc--------->".format(filename))
    pc_embed_all_2 = torch.stack(pc_embed_all_2)
    with open('ULIP_PointBERT_cads_feature_concate.pkl','wb') as f:
        pickle.dump((pc_embed_all_2,filename_list),f)

end_time = time.time()
duration = end_time - start_time
print("程序运行时间：", duration, "秒")