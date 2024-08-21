import os
import argparse
from tqdm import tqdm
import pdb
import random
import json

import cv2
import torch
import numpy as np
from models.locate import Net #as model
# from models.locate_sp import Net as NetSP
import torchvision.transforms as transforms
from PIL import Image

from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/home/gen/Project/aff_grounding/dataset/AGD20K/')
parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/AGD20K_llava/agd20k_llava_v4')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_preds')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
#### test
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

args = parser.parse_args()

if args.divide == "Seen":
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
elif args.divide == 'Unseen':
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]
elif args.divide == 'Generalization':
    aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                "talk_on", "text_on", "throw", "type_on", "wash", "write"]
else:
    raise ValueError


if args.divide == "Seen":
    args.num_classes = 36
elif args.divide == "Unseen":
    args.num_classes = 25
elif args.divide == "Generalization":
    args.num_classes = 36
else:
    raise ValueError

args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)


Tensor_to_Image = transforms.Compose([
    transforms.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    transforms.ToPILImage()
])


def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def evaluate():
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             #shuffle=False,
                                             shuffle=True,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = Net(aff_classes=args.num_classes).cuda()

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    for step, (image, label, mask_path) in enumerate(tqdm(TestLoader)):
        ego_pred = model.test_forward(image.cuda(), label.long().cuda())
        cluster_sim_maps = []
        ego_pred = np.array(ego_pred.squeeze().data.cpu())
        ego_pred = normalize_map(ego_pred, args.crop_size)

        names = mask_path[0].split("/")
        key = names[-3] + "_" + names[-2] + "_" + names[-1]
        GT_mask = GT_masks[key]
        GT_mask = GT_mask / 255.0

        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

        kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)
        KLs.append(kld)
        SIM.append(sim)
        NSS.append(nss)

        if args.viz:
            img_name = key.split(".")[0]
            viz_pred_test(args, image, ego_pred, GT_mask, aff_list, label, img_name)

    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    print(f"KLD = {round(mKLD, 3)}\nSIM = {round(mSIM, 3)}\nNSS = {round(mNSS, 3)}")


def visualize():
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=True,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = Net(aff_classes=args.num_classes).cuda()
    
    

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    # output_dir = 'outputs'
    output_dir = 'images'

    for step, (image, label, mask_path) in enumerate(tqdm(TestLoader)):
        for label_id, ps_label in enumerate(range(len(aff_list))):
            ps_label = torch.LongTensor([ps_label])
            #ego_pred = model.test_forward(image.cuda(), label.long().cuda())
            ego_pred = model.test_forward(image.cuda(), ps_label.long().cuda())
            cluster_sim_maps = []
            ego_pred = np.array(ego_pred.squeeze().data.cpu())
            ego_pred = normalize_map(ego_pred, args.crop_size)

            rgb = tensor_to_image(image[0])
            rgb = rgb[:, :, ::-1]
            output_path = os.path.join(output_dir, '{:0>6}_img.png'.format(step))
            Image.fromarray(rgb).save(output_path)
            rgb_gray = rgb.copy()
            alpha = 0.5
            heatmap_img = cv2.applyColorMap((ego_pred * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
            vis = cv2.addWeighted(heatmap_img, alpha, rgb_gray, 1 - alpha, 0)
            
            label_name = aff_list[ps_label[0]]
            output_path = os.path.join(output_dir, '{:0>6}_pred_{}.png'.format(step, label_name))
            Image.fromarray(vis).save(output_path)

        for i in range(3):    
            vis_empty = np.zeros_like(vis)
            output_path = os.path.join(output_dir, '{:0>6}_pred_zempty_{}.png'.format(step, i))
            # Image.fromarray(vis_empty).save(output_path)


def visualize_internet():
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=True,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = Net(aff_classes=args.num_classes).cuda()
    
    

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    # output_dir = 'outputs'
    output_dir = 'images'

    image_dir = '/home/shengyiq/datasets/afflm_gen/download'
    image_names = os.listdir(image_dir)
    image_pairs = {
        'Toilet_0.jpg': 'sit_on',
        'Toilet_2.jpg': 'sit_on',
        'Pliers_0.jpg': 'hold',
    }

    for step, image_name in enumerate(image_pairs):
        action = image_pairs[image_name]

        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        
        image = testset.transform(image).unsqueeze(0)
        # breakpoint()

        ps_label = torch.LongTensor([aff_list.index(action)])
        ego_pred = model.test_forward(image.cuda(), ps_label.long().cuda())

        cluster_sim_maps = []
        ego_pred = np.array(ego_pred.squeeze().data.cpu())
        ego_pred = normalize_map(ego_pred, args.crop_size)

        rgb = tensor_to_image(image[0])
        rgb = rgb[:, :, ::-1]
        output_path = os.path.join(output_dir, '{:0>6}_img.png'.format(step))
        Image.fromarray(rgb).save(output_path)
        rgb_gray = rgb.copy()
        alpha = 0.5
        heatmap_img = cv2.applyColorMap((ego_pred * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
        vis = cv2.addWeighted(heatmap_img, alpha, rgb_gray, 1 - alpha, 0)
        
        label_name = aff_list[ps_label[0]]
        output_path = os.path.join(output_dir, '{:0>6}_pred_{}.png'.format(step, action))
        Image.fromarray(vis).save(output_path)


def visualize_llava():
    set_seed(seed=0)

    from data.datatest import TestData

    testset = TestData(
        image_root=args.test_root,
        crop_size=args.crop_size,
        divide=args.divide, 
        mask_root=args.mask_root
    )

    # agd_dir = '/home/ubuntu/efs/AGD20K'
    agd_dir = '/home/shengyiq/datasets/AGD20K'

    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir, '..', 'images')
    split = 'val'
    # split = 'train'
    f = open(os.path.join(dataset_dir, 'data_{}.json'.format(split))) 
    data = json.load(f)
    random.seed(2023)
    random.shuffle(data)

    model = Net(aff_classes=args.num_classes).cuda()
    # model = NetSP(aff_classes=args.num_classes).cuda()

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(args.divide + "_gt.t7")

    # output_dir = 'outputs'
    output_dir = 'images'

    # for step, (image, label, mask_path) in enumerate(tqdm(TestLoader)):
    for step, entry in enumerate(tqdm(data)):
        img_path = os.path.join(image_dir, entry['image'])

        # breakpoint()

        image = testset.load_img(img_path).unsqueeze(0)
        
        # img_name = '{}_img.png'.format(entry['id'])
        # image.save(os.path.join(output_img_dir, img_name))
        # question = entry['conversations'][0]['value']
        # answer = entry['conversations'][1]['value']
        # qs = question.split('\n')[0]
        
        gt_path = os.path.join(agd_dir, entry['gt_path'])
        label_str = gt_path.split('/')[-3]

        # pdb.set_trace()

        label = torch.LongTensor([aff_list.index(label_str)])
        aff_gt = np.array(Image.open(gt_path))
        aff_gt = cv2.resize(aff_gt, (224, 224))
        vis_name = entry['id']
        

        ps_label = label
        # pdb.set_trace()
        #ego_pred = model.test_forward(image.cuda(), label.long().cuda())
        ego_pred = model.test_forward(image.cuda(), ps_label.long().cuda())
        cluster_sim_maps = []
        ego_pred = np.array(ego_pred.squeeze().data.cpu())
        ego_pred = normalize_map(ego_pred, args.crop_size)

        rgb = tensor_to_image(image[0])
        rgb = rgb[:, :, ::-1]
        # output_path = os.path.join(output_dir, '{:0>6}_0_img.png'.format(step))
        output_path = os.path.join(output_dir, entry['image'].replace('.png', '_image.png'))
        Image.fromarray(rgb).save(output_path)
        rgb_gray = rgb.copy()
        alpha = 0.5
        heatmap_img = cv2.applyColorMap((ego_pred * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
        vis = cv2.addWeighted(heatmap_img, alpha, rgb_gray, 1 - alpha, 0)
        
        label_name = aff_list[ps_label[0]]
        # output_path = os.path.join(output_dir, '{:0>6}_1_pred_{}.png'.format(step, label_name))
        output_path = os.path.join(output_dir, entry['image'].replace('.png', '_pred.png'))
        Image.fromarray(vis).save(output_path)

        Image.fromarray(vis).save(os.path.join(output_dir, entry['image']))

        # gt
        # output_path = os.path.join(output_dir, '{:0>6}_2_gt_{}.png'.format(step, label_name))
        output_path = os.path.join(output_dir, entry['image'].replace('.png', '_gt.png'))
        # aff_gt = np.array(Image.open(mask_path[0]))
        # aff_gt = cv2.resize(aff_gt, (args.crop_size, args.crop_size))
        rgb_gt = rgb.copy()
        heatmap_img = cv2.applyColorMap(aff_gt.astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
        vis = cv2.addWeighted(heatmap_img, alpha, rgb_gt, 1 - alpha, 0)
        Image.fromarray(vis).save(output_path)


if __name__=='__main__':
    # visualize()
    visualize_llava()
    # evaluate()