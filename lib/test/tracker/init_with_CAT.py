import math
import importlib
import argparse
from lib.models.CATos import build_cat_os
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import cv2
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.test.tracker.test_dataset_info import _get_sequence_info_list, get_seq_pth, init_resize
from lib.test.tracker.init_data_process import click_sample, box_iou

dataset_root = {
    'lasot': '/media/space/T7/LaSOT/dataset/images',
    'lasot_extension_subset': '/data3/LaSOT_extension_subset',
    'trackingnet': '/data1/Datasets/TrackingNet/TEST/frames',
    'got10k': '/data1/Datasets/GOT-10k/test',
    'uav123': '/media/space/T1/UAV123',
    'uav112': '/media/space/T1/UAVTrack112',
    'dtb70': '/media/space/T7/DTB70',
}

class CAT(BaseTracker):
    def __init__(self, params):
        super(CAT, self).__init__(params)
        network = build_cat_os(params.cfg, training=False)
        ck = torch.load(self.params.checkpoint, map_location='cpu')
        network.load_state_dict(ck, strict=True)
        print('laoding new params')
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.params.search_size_s2 // self.cfg.MODEL.BACKBONE.STRIDE

        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()


    def initialize(self, dataset):
        dataset_pth = dataset_root[dataset]  # Update your dataset path here.
        test_seqs = _get_sequence_info_list(dataset)

        iou_all = torch.tensor([0], dtype=torch.float32).cuda()
        iou_all_2 = torch.tensor([0], dtype=torch.float32).cuda()  # cal_IOU

        for i in range(len(test_seqs)):
            seq = test_seqs[i]
            img_pth, gt_pth, gt = get_seq_pth(dataset, dataset_pth, seq)

            image = cv2.imread(img_pth)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if dataset != 'uav112':
                try:
                    gt = np.loadtxt(gt_pth, delimiter=',', dtype=np.float64)
                except:
                    gt = np.loadtxt(gt_pth, delimiter=None, dtype=np.float64)
            else:
                gt = gt
            first_anno = gt.tolist() if dataset in ['trackingnet', 'got10k'] else gt[0].tolist()  # ndarray(4)

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
                print(seq, 's1: ', iou)
                iou_all+=iou

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
                                                            out_dict_2s[-1]['offset_map'], int(self.params.search_size_s2/16))
                pred_boxes = pred_boxes.view(-1, 4)
                # Baseline: Take the mean of all pred boxes as the final result
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size_s2 / resize_factor_x).tolist()  # (cx, cy, w, h) [0,1]
                # get the final box result
                self.state = init_box_s1[0]
                final_bbox = self.map_box_back(pred_box, resize_factor_x)
                init_box_s2 = torch.tensor([[final_bbox[0].item(), final_bbox[1].item(), final_bbox[0].item()+final_bbox[2], final_bbox[1].item()+final_bbox[3]]])
                iou_2, _ = box_iou(first_anno, init_box_s2.cuda())
                gt[0] = np.array(torch.tensor([[final_bbox[0].item(), final_bbox[1].item(), final_bbox[2], final_bbox[3]]])).reshape(4)
                # np.savetxt(sv_pth_file2, gt, fmt='%.2f', delimiter=",")
                iou_all_2 += iou_2
                print(seq, 's2: ', iou_2)

        print('Avg.IOU for stage1: ', iou_all.item()/len(test_seqs))
        print('Avg.IOU for stage2: ', iou_all_2.item()/len(test_seqs))


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
    return CAT



def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='lasot',
                    help='Name of dataset(lasot, lasot_extension_subset, trackingnet, got10k, uav123, uav112, dtb70).')

    args = parser.parse_args()

    param_module = importlib.import_module('lib.test.parameter.{}'.format('CATos'))
    params = param_module.parameters(args.tracker_param)
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format(args.tracker_name))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params)
    tracker.initialize(args.dataset_name)

if __name__ == '__main__':
    main()