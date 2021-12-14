
import argparse
import base64
import json
from logging import root
import os

from PIL import Image
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi
from utils import fast_hist, label_to_color_image


def sdk_args():
    parser = argparse.ArgumentParser("Auto-DeepLab SDK Inference")
    parser.add_argument('--pipeline', type=str, default='', help='path to pipeline file')
    # val data
    parser.add_argument('--data_root', type=str, default='',
                        help='root path of val data')
    parser.add_argument('--result_path', type=str, default='',
                        help='output_path')
    parser.add_argument('--num_classes', type=int, default=19,
                        help='number of classes')
    args = parser.parse_args()
    return args


class CityscapesDataLoader(object):
    def __init__(self, root, split='val', ignore_label=255):
        super(CityscapesDataLoader, self).__init__()
        images_base = os.path.join(root, 'leftImg8bit', split)
        annotations_base = os.path.join(root, 'gtFine', split)
        self.img_id = []
        self.img_path = []
        self.img_name = []
        self.images = []
        self.gtFiles = []
        for root, _, files in os.walk(images_base):
            for filename in files:
                if filename.endswith('.png'):
                    self.img_path.append(root)
                    self.img_name.append(filename)
                    folder_name = root.split(os.sep)[-1]
                    gtFine_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    _img = os.path.join(root, filename)
                    _gt = os.path.join(annotations_base, folder_name, gtFine_name)
                    self.images.append(_img)
                    self.gtFiles.append(_gt)
        self.len = len(self.images)
        self.cur_index = 0
        print(f"found {self.cur_index} images")

    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self.cur_index == self.len:
            raise StopIteration()
        
        with open(self.images[self.cur_index], 'rb') as f:
            image = f.read()
        gtFine = Image.open(self.gtFiles[self.cur_index])
        gtFine = np.array(gtFine).astype(np.uint8)
        dataItem = {
            'file_path': self.img_path[self.cur_index],
            'file_name': self.img_name[self.cur_index],
            'img': image,
            'gt': gtFine,
        }
        self.cur_index += 1
        return dataItem



def _init_stream(pipeline_path):
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        raise RuntimeError(f"Failed to init stream manager, ret={ret}")

    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()

        ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise RuntimeError(f"Failed to create stream, ret={ret}")
        return stream_manager_api


def _do_infer(stream_manager_api, data_input):
    stream_name = b'segmentation'
    unique_id = stream_manager_api.SendDataWithUniqueId(
        stream_name, 0, data_input)
    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")

    timeout = 6000
    infer_result = stream_manager_api.GetResultWithUniqueId(
        stream_name, unique_id, timeout)
    if infer_result.errorCode != 0:
        raise RuntimeError(
            "GetResultWithUniqueId error, errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))

    load_dict = json.loads(infer_result.data.decode())
    image_mask = load_dict["MxpiImageMask"][0]
    data_str = base64.b64decode(image_mask['dataStr'])
    shape = image_mask['shape']
    return np.frombuffer(data_str, dtype=np.uint8).reshape(shape)

def main():
    """main"""
    args = sdk_args()

    # init stream manager
    stream_manager_api = _init_stream(args.pipeline)
    if not stream_manager_api:
        exit(1)
    
    os.makedirs(args.result_path, exist_ok=True)
    data_input = MxDataInput()
    dataset = CityscapesDataLoader(args.data_root)
    hist = np.zeros((args.num_classes, args.num_classes))
    for data_item in dataset:
        print(f"start infer {data_item['file_name']}")
        data_input.data = data_item['img']
        gtFine = data_item['gt']
        pred = _do_infer(stream_manager_api, data_input)

        hist += fast_hist(pred.flatten(), gtFine.flatten(), args.num_classes)
        color_mask_res = label_to_color_image(pred)

        folder_path = os.path.join(args.result_path, data_item['file_path'].split(os.sep)[-1])
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        result_file = os.path.join(folder_path, data_item['file_name'].replace('leftImg8bit', 'pred_color'))
        color_mask_res.save(result_file)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("per-class IOU", iou)
    print("mean IOU", np.nanmean(iou))

    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    main()
