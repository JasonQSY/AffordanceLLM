import os
# from scipy.ndimage import maximum_filter
import pdb
from PIL import Image
import numpy as np
# from scipy.ndimage import label
import json
from tqdm import tqdm
import random
import yaml
import torch


from agd20k.datatrain import TrainData
from agd20k.datatest import TestData
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from fastchat.model import load_model, get_conversation_template, add_model_args



def ask_vicuna(model, tokenizer, msg):
    # model_path = 'lmsys/vicuna-33b-v1.3'
    model_path = 'lmsys/vicuna-13b-v1.3'
    temperature = 0.7
    repetition_penalty = 1.0
    max_new_tokens = 512

    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    return outputs


def main():
    random.seed(6)

    data_root = '/home/ubuntu/efs/AGD20K'
    output_img_dir = '/home/ubuntu/efs/AGD20K_llava/images'
    output_dir = '/home/ubuntu/efs/AGD20K_llava'

    dataset_name = 'agd20k_llava_v16_normal'
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    divide = 'Seen'
    # divide = 'Unseen'
    crop_size = 224
    resize_size = 256

    exocentric_root = os.path.join(data_root, divide, "trainset", "exocentric")
    egocentric_root = os.path.join(data_root, divide, "trainset", "egocentric")
    test_root = os.path.join(data_root, divide, "testset", "egocentric")
    mask_root = os.path.join(data_root, divide, "testset", "GT")

    # split v1
    full_aff_list = [
        'beat', "boxing", "brush_with", "carry", "catch",
        "cut", "cut_with", "drag", 'drink_with', "eat",
        "hit", "hold", "jump", "kick", "lie_on", "lift",
        "look_out", "open", "pack", "peel", "pick_up",
        "pour", "push", "ride", "sip", "sit_on", "stick",
        "stir", "swing", "take_photo", "talk_on", "text_on",
        "throw", "type_on", "wash", "write"
    ]
    seen_aff_list = [
        "carry", "catch", "cut", "cut_with", 'drink_with',
        "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
        "pick_up", "pour", "push", "ride", "sip", 
    ]
    unseen_aff_list = [
        'beat', "boxing", "brush_with", "drag", "lift", "look_out",
        "pack", "stir", "talk_on", "text_on", "write", "sit_on", "stick",
        "swing", "take_photo", "throw", "type_on", "wash",
    ]
    full_obj_list = [
        'apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
        'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
        'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
        'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
        'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
        'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
        'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
        'tennis_racket', 'toothbrush', 'wine_glass'
    ]
    seen_obj_list = [
        'apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
        'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
        'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
        'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
    ]
    unseen_obj_list = [
        'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
        'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
        'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
        'tennis_racket', 'toothbrush', 'wine_glass',
    ]
    locate_seen_obj_list = [
        'motorcycle', 'baseball_bat', 'tennis_racket', 'badminton_racket', 'frisbee', 'baseball', 'javelin', 'rugby_ball', 'discus', 'skateboard', 'surfboard', 'snowboard', 'bench', 'couch', 'chair', 'wine_glass', 'orange', 'apple', 'carrot', 'punching_bag', 'oven', 'bottle', 'suitcase', 'microwave', 'book', 'hot_dog', 'keyboard', 'fork', 'cell_phone', 'hammer', 'bowl', 'toothbrush', 'scissors'
    ]
    locate_unseen_obj_list = [
        'bicycle', 'golf_clubs', 'basketball', 'skis', 'bed', 'cup', 'banana', 'soccer_ball', 'refrigerator', 'broccoli', 'laptop', 'knife', 'camera', 'axe',
    ]

    print(seen_obj_list)
    print(unseen_obj_list)

    trainset = TestData(
        image_root=test_root,
        crop_size=crop_size,
        divide=divide,
        mask_root=mask_root,
        aff_list=full_aff_list,
        # aff_list=seen_aff_list,
        # obj_list=full_obj_list,
        obj_list=seen_obj_list,
        # obj_list=locate_seen_obj_list,
    )

    testset = TestData(
        image_root=test_root,
        crop_size=crop_size,
        divide=divide,
        mask_root=mask_root,
        aff_list=full_aff_list,
        # aff_list=unseen_aff_list,
        # obj_list=full_obj_list,
        obj_list=unseen_obj_list,
        # obj_list=locate_unseen_obj_list,
    )

    print(len(trainset))
    print(len(testset))

    datasets = {
        'train': trainset,
        'val': testset,
        'test': testset,
    }

    train_imgs = {}

    for split in ('train', 'val'):
        dataset = datasets[split]
        llava_gt = []

        for idx, triplet in enumerate(tqdm(dataset)):
            image = triplet[0]
            action_label = triplet[1]
            gt_path = triplet[2]

            # exclude train images
            adg_img_name = gt_path.split('/')[-1]
            if split == 'train':
                train_imgs[adg_img_name] = True
            else:
                if adg_img_name in train_imgs:
                    continue

            aff_gt = np.array(Image.open(gt_path))

            # save image
            splits = gt_path.split('/')
            image_name = '_'.join(splits[-3:])
            obj_name = splits[-2]
            rel_gt_path = '/'.join(splits[-6:])

            rel_depth_path = image_name.replace('.png', '-dpt_beit_large_512.pfm')
            rel_depth_path = os.path.join('..', 'depth', rel_depth_path)

            action_label_clean = action_label.replace('_', ' ')
            obj_str_clean = obj_name.replace('_', ' ')

            if split == 'train':
                # action_strs = ['interact', action_label_clean]
                # obj_strs = ['object', obj_str_clean]
                action_strs = [action_label_clean]
                obj_strs = [obj_str_clean]
            elif split == 'val':
                action_strs = [action_label_clean]
                obj_strs = [obj_str_clean]
            else:
                raise ValueError

            action_str = random.choice(action_strs)
            obj_str = random.choice(obj_strs)

            question = "What part of the {} should we interact with in order to {} it?".format(obj_str, action_str)
            
            # ask language model
            # if action_label_clean in lm_insts and obj_str_clean in lm_insts[action_label_clean]:
            #     answer_vicuna = lm_insts[action_label_clean][obj_str_clean]
            # else:
            #     question_vicuna = question + " Please use plain language and keep the instruction simple."
            #     answer_vicuna = ask_vicuna(model, tokenizer, question_vicuna)
            #     if action_label_clean not in lm_insts:
            #         lm_insts[action_label_clean] = {}
            #     lm_insts[action_label_clean][obj_str_clean] = answer_vicuna

            lm_inst = lm_insts[action_label_clean][obj_str_clean]
            # if len(lm_inst) > 0:
            #     question += ' ' + 'You should interact with the ' + lm_inst + '.'

            question = '<image>\n' + question

            action_str = random.choice(action_strs)
            obj_str = random.choice(obj_strs)

            answer_text = "You can {} the highlighted area. <seg_patch>".format(action_label_clean)


            image.save(os.path.join(output_img_dir, image_name))

            llava_triplet = {
                "id": image_name.split('.')[0],
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": question,
                    },
                    {
                        "from": "agd20k",
                        "value": answer_text,
                    }
                ],
                'depth': rel_depth_path,
                'gt_path': rel_gt_path,
            }
            llava_gt.append(llava_triplet)

        f = open(os.path.join(output_dir, dataset_name, 'data_{}.json'.format(split)), 'w')
        json.dump(llava_gt, f)
        f.close()

    # save all lm instructions for future reference
    f = open(os.path.join(output_dir, dataset_name, 'lm_insts.json'), 'w')
    json.dump(lm_insts, f)
    f.close()

    dataset_record = {
        'seen_aff_list': seen_aff_list,
        'unseen_aff_list': unseen_aff_list,
    }
    f = open(os.path.join(output_dir, dataset_name, 'dataset_records.json'), 'w')
    json.dump(dataset_record, f)
    f.close()


if __name__=='__main__':
    main()
