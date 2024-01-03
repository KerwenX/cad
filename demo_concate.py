import argparse
import os
from collections import OrderedDict
import math
import time

import numpy as np
import wandb
from typing import List
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

from data.dataset_3d import *

import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import datetime
import warnings
import logging
log_format = '%(asctime)s - DEMO - Concate - %(levelname)s - %(message)s'
logging.basicConfig(filename='demo_demo_concate.out.log', level=logging.INFO, filemode='a',format=log_format)
logging.getLogger().addHandler(logging.StreamHandler())
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
    parser.add_argument('--test_ckpt_addr', default='/home/aston/Desktop/python/CAD-Matching/ckpts/concate/checkpoint_4901.pt', help='the ckpt to test 3d zero shot')
    # parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    return parser

# --test_ckpt_addr /home/aston/Downloads/checkpoint_best_0628.pt
def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def main(args):
    total_file_number = 0
    image_correct_top1 = 0
    image_correct_top5 = 0
    text_correct_top1 = 0
    text_correct_top5 = 0
    common_correct = 0
    cat_image_acc_top1 = {}

    taxonomy_file = 'data/ROCA/taxonomy.json'
    with open(taxonomy_file, 'rb') as f:
        taxonomies = json.load(f)

    # pc_encoder
    with open('ULIP_PointBERT_cads_feature_concate.pkl', 'rb') as f:
        pc_embed_all, filename_list = pickle.load(f)

    # image
    image_dir = 'data/ROCA/rendered_images'
    dirname_list = os.listdir(image_dir)

    pc_file = 'data/ROCA/roca_pc'
    pc_encoder_label = False

    # model setting
    tokenizer = SimpleTokenizer()

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

    logit_scale = torch.tensor(76.0, device='cuda')

    logging.getLogger().addHandler(logging.StreamHandler())
    with torch.no_grad():
        for dirname in dirname_list:
            image_path_list = os.listdir(os.path.join(image_dir,dirname))
            # ['color','depth','albedo','mask']
            for image_path in image_path_list:
                if "depth" in image_path or "albedo" in image_path or 'mask' in image_path:
                    continue
                total_file_number+=1
                image_path = os.path.join(image_dir,dirname,image_path)

                # image load
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    transforms.ToTensor(),
                    normalize
                ])
                # image_path = 'data/ROCA/rendered_images/02747177-75dbeb01fdc7f265d0a96520c31993ad/02747177-75dbeb01fdc7f265d0a96520c31993ad_0000.png'
                img = Image.open(image_path)
                img = img.convert('RGB')
                img = train_transform(img).unsqueeze(dim=0).cuda()

                # text/label load
                catid_cad = image_path.split('/')[-1].split('-')[0]
                id_cad = image_path.split('/')[-1].split('_')[0].split('-')[1]
                thing_classes = []
                tokenized_captions = []
                for tax in taxonomies:
                    if tax['synsetId']==catid_cad:
                        thing_classes = [cat for cat in tax['name'].split(',')]
                caption = thing_classes[0]
                tokenized_captions.append(tokenizer(caption))
                tokenized_captions = torch.stack(tokenized_captions).cuda()

                # text encoder
                text_embed_all = []

                text_for_one_sample = tokenized_captions
                text_embed = model.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

                text_embed = torch.stack(text_embed_all)

                # image encoder
                image_embed = model.encode_image(img)

                concate_embed = (3 * image_embed + text_embed) / 4

                # normalized features
                pc_embed_all = F.normalize(pc_embed_all, dim=-1, p=2)
                # text_embed = F.normalize(text_embed, dim=-1, p=2)
                # image_embed = F.normalize(image_embed, dim=-1, p=2)
                concate_embed = F.normalize(concate_embed, dim=-1, p=2)



                # cosine similarity as logits
                # logits_per_pc_text = logit_scale * pc_embed_all @ text_embed.t()
                # logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
                # logits_per_pc_image = logit_scale * pc_embed_all @ image_embed.t()
                # logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()
                logits_per_concate_pc = logit_scale * concate_embed @ pc_embed_all.t()
                # logits_per_pc_concate = logit_scale * pc_embed_all @ concate_embed.t()


                # _,image_retrival = torch.topk(logits_per_image_pc,k=1)
                # _,image_retrival_top5 = torch.topk(logits_per_image_pc,k=5)
                # _,text_retrival = torch.topk(logits_per_text_pc,k=1)
                # _,text_retrival_top5 = torch.topk(logits_per_text_pc,k=5)
                _, concate_retrival = torch.topk(logits_per_concate_pc, k=1)
                _, concate_retrival_top5 = torch.topk(logits_per_concate_pc, k=5)

                # image_pc_retrieval_namelist = [filename_list[index] for index in image_retrival_top5[0]]
                # text_pc_retrieval_namelist = [filename_list[index] for index in text_retrival_top5[0]]
                concate_pc_retrieval_namelist = [filename_list[index] for index in concate_retrival_top5[0]]

                # image_pc_catid_retre = [os.path.splitext(name)[0].split('-')[0] for name in image_pc_retrieval_namelist]
                # text_pc_catid_retre = [os.path.splitext(name)[0].split('-')[0] for name in text_pc_retrieval_namelist]
                # image_pc_idcad_retre = [os.path.splitext(name)[0].split('-')[1] for name in image_pc_retrieval_namelist]
                # text_pc_idcad_retre = [os.path.splitext(name)[0].split('-')[1] for name in text_pc_retrieval_namelist]
                concate_pc_idcad_retre = [os.path.splitext(name)[0].split('-')[1] for name in concate_pc_retrieval_namelist]

                if caption not in cat_image_acc_top1.keys():
                    cat_image_acc_top1[caption]=0
                    cat_image_acc_top1[caption+'_files_num']=0
                cat_image_acc_top1[caption+'_files_num']+=1

                if id_cad in concate_pc_idcad_retre:
                    image_correct_top5+=1
                    if id_cad == concate_pc_idcad_retre[0]:
                        image_correct_top1+=1
                        cat_image_acc_top1[caption]+=1

                # common_id = set(image_pc_idcad_retre).intersection(text_pc_idcad_retre)


                if total_file_number%50==0:
                    # print(" [{}]".format(datetime.datetime.now()), end='')
                    logging.info(
                        "[{}-{}][{}] ".format(catid_cad, id_cad, caption)+
                        "Image_top1_acc: {:.3f}%, Image_top5_acc: {:.3f}%. ".format(
                              image_correct_top1 * 100 / total_file_number,
                              image_correct_top5 * 100 / total_file_number,
                              # text_correct_top1 * 100 / total_file_number,
                              # text_correct_top5 * 100 / total_file_number,
                          )
                    )

    logging.info(
        "[RESULT] Image_top1_acc: {}%, Image_top5_acc: {}%".format(
            image_correct_top1*100/total_file_number,
            image_correct_top5*100/total_file_number,
            # text_correct_top1*100/total_file_number,
            # text_correct_top5*100/total_file_number,
        )
    )
    logging.info(" -----> number of file is {}".format(total_file_number))

    for key in cat_image_acc_top1.keys():
        if "files_num" in key:
            continue
        logging.info(
            "{}: correct files' number is {}/total files' number is {}, acc is {:.3f}%".format(
                key,
                cat_image_acc_top1[key],
                cat_image_acc_top1[key+'_files_num'],
                cat_image_acc_top1[key]*100/cat_image_acc_top1[key+'_files_num'],
            )
        )

# retrieval CAD with image and text respectively
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('ULIP demo', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    duration = end_time - start_time
    print("程序运行时间：", duration, "秒")