import math

from lib.models.CATos import build_cat_os
from lib.test.tracker.basetracker import BaseTracker
import torch
import random
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.test.tracker.init_data_process import click_sample, box_iou
from lib.test.tracker.test_dataset_info import init_resize

class CATos(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CATos, self).__init__(params)
        self.test = True
        self.iou = torch.tensor([0], dtype=torch.float64)
        network = build_cat_os(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        print('laoding new params')
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.lang = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE_s2 // self.cfg.MODEL.BACKBONE.STRIDE
        a = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)  #  ** 1.6
        self.output_window = a.cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.imgs = []
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict, load_init=0, init_select=None):

        first_anno = info['init_bbox']

        h, w = image.shape[0:-1]
        mask_points_ori, _ = click_sample(torch.tensor(first_anno), [h, w], self.params.search_size_s1)

        x_patch_arr, x_bbox, x_amask_arr, mask_pad, mask_point, map_param = init_resize(image,
                                                                                        self.params.search_size_s1,
                                                                                        torch.tensor(first_anno),
                                                                                        pr_points=mask_points_ori)
        x_patch_arr1 = self.preprocessor.process(x_patch_arr[0], x_amask_arr[0])
        x = x_patch_arr1.tensors

        prompt_id = mask_point.unsqueeze(0).cuda()
        with torch.no_grad():
            out_dict = self.network(search=[x], pro_id=prompt_id)
            pred_box_s1 = out_dict[0]['pred_boxes'][0] * self.params.search_size_s1

            first_anno = torch.tensor(first_anno).cuda().reshape(1, 4)
            first_anno[:, 2:] += first_anno[:, :2]  # (x, y, w, h) -> (x1, y1, x2, y2)
            pred_box_s1 = self.mapback(pred_box_s1, map_param)  # tensor 1, 4

            init_box_s1 = pred_box_s1.clone().detach().cpu()
            init_box_s1[:, 2:] -= init_box_s1[:, :2]

            iou, _ = box_iou(first_anno, pred_box_s1)
            print('s1: ', iou)

            z_patch_arr, resize_factor_z, z_amask_arr = sample_target(image, init_box_s1.tolist()[0],
                                                                      self.params.template_factor,
                                                                      output_sz=self.params.template_size)
            x_patch_arr, resize_factor_x, x_amask_arr = sample_target(image, init_box_s1.tolist()[0],
                                                                      self.params.search_factor_s2,
                                                                      output_sz=self.params.search_size_s2)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)
            sr = self.preprocessor.process(x_patch_arr, x_amask_arr)
            pr_s2 = self.prompt_extract_stage2(init_box_s1[0], resize_factor_x)
            box_prompt = torch.from_numpy(np.stack([x.numpy() for x in [pr_s2]])[None, :, :])
            prompt_id = self.refine_prompt(box_prompt.clone().detach().cpu())

            out_dict_2s = self.network.stage2(temp=[template.tensors], search=[sr.tensors], prompt=prompt_id)
            pred_score_map = out_dict_2s[-1]['score_map']
            response = self.output_window * pred_score_map  # win 0.5

            pred_boxes = self.network.box_head.cal_bbox(response, out_dict_2s[-1]['size_map'],
                                                        out_dict_2s[-1]['offset_map'],
                                                        int(self.params.search_size_s2 / 16))
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size_s2 / resize_factor_x).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = init_box_s1[0]
            final_bbox = self.map_box_back(pred_box, resize_factor_x)
            init_box_s2 = torch.tensor([[final_bbox[0].item(), final_bbox[1].item(),
                                         final_bbox[0].item() + final_bbox[2],
                                         final_bbox[1].item() + final_bbox[3]]])
            iou_2, _ = box_iou(first_anno, init_box_s2.cuda())
            info['init_bbox'] = [final_bbox[0].item(), final_bbox[1].item(), final_bbox[2], final_bbox[3]]
            print('s2: ', iou_2)

        print('Init.IOU: ', iou_2.item())

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor_s2,
                                                                output_sz=self.params.search_size_s2)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.stage2(
                temp=[self.z_dict1.tensors], search=[x_dict.tensors], tracking_st=True, mode='track')

        # add hann windows
        pred_score_map = out_dict[-1]['score_map']

        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict[-1]['size_map'], out_dict[-1]['offset_map'], self.feat_sz)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size_s2 / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        return {"target_bbox": self.state}

    def refine_prompt(self, anno):  # 1, 64 , 4
        N, B, _ = anno.shape
        len_pro = 23 if self.params.search_size_s2 == 384 else 15
        anno = anno.unsqueeze(-2).repeat(1, 1, len_pro, 1).numpy()
        anno = torch.from_numpy(np.stack([anno[:, :, :, 0:2], anno[:, :, :, 2:4]], axis=2)).cuda()
        return anno.reshape(N, B, -1, 2).contiguous().float()

    def prompt_extract_stage2(self, box_extract: torch.Tensor, resize_factor: float) -> torch.Tensor:
        search_region_size = torch.Tensor([self.params.search_size_s2, self.params.search_size_s2])
        search_region_center = (search_region_size - 1) / 2
        scaled_box_wh = box_extract[2:4] * resize_factor
        box_in_search_region = torch.cat((search_region_center, scaled_box_wh))
        return box_in_search_region / search_region_size[0]


    def mapback(self, box, map_param):
        box = box / map_param['fac']
        x, y, w, h = box[0].clone()
        box[:, 2] = box[:, 0] + 0.5* w
        box[:, 3] = box[:, 1] + 0.5* h
        box[:, 0] = box[:, 0] - 0.5* w
        box[:, 1] = box[:, 1] - 0.5* h

        return box


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size_s2 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]



def get_tracker_class():
    return CATos
