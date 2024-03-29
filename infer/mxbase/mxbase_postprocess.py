"""Evaluate mIOU and Pixel accuracy"""
import os
import argparse
import ast
import PIL

import cv2
from PIL import Image
import numpy as np


def fast_hist(predict, label, n):
    """
    fast_hist
    inputs:
        - predict (ndarray)
        - label (ndarray)
        - n (int) - number of classes
    outputs:
        - fast histogram
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(np.int32) + predict[k], minlength=n ** 2).reshape(n, n)

def encode_segmap(lbl, ignore_label):
    """encode segmap"""
    mask = np.uint8(lbl)

    num_classes = 19
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    class_map = dict(zip(valid_classes, range(num_classes)))
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_label
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]

    return mask

def decode_segmap(pred):
    """decode_segmap"""
    mask = np.uint8(pred)

    num_classes = 19
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    rank_classes = range(num_classes)

    class_map = dict(zip(rank_classes, valid_classes))

    for _rank in rank_classes:
        mask[mask == _rank] = class_map[_rank]

    return mask

def get_color(npimg):
    """get_color"""
    cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]
    img = Image.fromarray(npimg.astype('uint8'), "P")
    img.putpalette(cityspallete)
    out_img = np.array(img.convert('RGB'))
    return out_img

def infer(args):
    """infer"""
    images_base = os.path.join(args.dataset_path, 'leftImg8bit/val')
    annotations_base = os.path.join(args.dataset_path, 'gtFine/val')
    hist = np.zeros((args.num_classes, args.num_classes))
    for root, _, files in os.walk(images_base):
        for filename in files:
            if filename.endswith('.png'):
                print("start post ", filename)
                file_name = filename.split('.')[0]

                pred_file = os.path.join(args.result_path, file_name + "_MxBase_infer.png")

                pred = np.array(Image.open(pred_file), dtype=np.uint8)
                folder_name = root.split(os.sep)[-1]

                if args.cal_acc:
                    gtFine_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    label_file = os.path.join(annotations_base, folder_name, gtFine_name)
                    label = np.array(cv2.imread(label_file, cv2.IMREAD_GRAYSCALE), np.uint8)
                    label = encode_segmap(label, 255)
                    hist = hist + fast_hist(pred.copy().flatten(), label.flatten(), args.num_classes)

                if args.save_img:

                    # colorful segmentation image
                    colorImg_name = filename.replace('leftImg8bit', 'predImg_colorful')
                    colorImg_root = args.output_path
                    colorImg_root = os.path.join(colorImg_root.replace('output', 'output_img'), folder_name)
                    colorImg_file = os.path.join(colorImg_root, colorImg_name)
                    if not os.path.isdir(colorImg_root):
                        os.makedirs(colorImg_root)
                    color_pred = get_color(pred.copy())
                    color_pred = cv2.cvtColor(np.asarray(color_pred), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(colorImg_file, color_pred, [cv2.IMWRITE_PNG_COMPRESSION])

    if args.cal_acc:
        miou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist) + 1e-10)
        miou = round(np.nanmean(miou) * 100, 2)
        print("mIOU = ", miou, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-DeepLab Inference post-process")
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path for evaluation")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--result_path", type=str, default="", help="Pred png file path.")
    parser.add_argument("--output_path", type=str, default="", help="Output path.")
    parser.add_argument("--save_img", type=ast.literal_eval, default=True, help="Whether save pics after inference.")
    parser.add_argument("--cal_acc", type=ast.literal_eval, default=True, help="Calculate mIOU or not.")
    Args = parser.parse_args()
    infer(Args)
