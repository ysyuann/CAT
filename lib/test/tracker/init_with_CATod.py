
import importlib
import argparse
from lib.models.CATod import build_cat_od
from lib.test.tracker.basetracker import BaseTracker
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate, generate_point_cond
import torch
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.test_dataset_info import get_seq_pth, init_resize, get_test_seq_list
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


class CATod(BaseTracker):
    def __init__(self, params):
        super(CATod, self).__init__(params)
        self.test = True
        network = build_cat_od(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        print('laoding new params')
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz,self.feat_sz]).long(), centered=True).cuda()

    def initialize(self, dataset):
        dataset_pth = dataset_root[dataset]  # Update your dataset path here.
        test_seqs = get_test_seq_list(dataset, dataset_pth)

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
            mask_points_ori, _ = click_sample(torch.tensor(first_anno), [h, w], self.params.search_size, prompt_len=49)  # tensor 30, 2

            x_patch_arr, x_bbox, x_amask_arr, mask_pad, mask_point, map_param = init_resize(image, int(self.params.search_size), torch.tensor(first_anno), pr_points = mask_points_ori)
            x_patch_arr1 = self.preprocessor.process(x_patch_arr[0], x_amask_arr[0])
            x = x_patch_arr1.tensors

            prompt_id = mask_point.unsqueeze(0).cuda()
            with torch.no_grad():

                box_mask_z, ce_keep_rate = self.ce(1)

                out_dict = self.network(template=[None], search=[x], pro_id = prompt_id, s=1,
                             ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,)
                pred_box_s1 = out_dict[0]['pred_boxes'][0]*self.params.search_size

                first_anno = torch.tensor(first_anno).cuda().reshape(1, 4)
                first_anno[:, 2:] += first_anno[:, :2]  # (x, y, w, h) -> (x1, y1, x2, y2)

                pred_box_s1 = self.mapback(pred_box_s1, map_param)

                init_box_s1 = pred_box_s1.clone().detach().cpu()
                init_box_s1[:, 2:] -= init_box_s1[:, :2]

                iou, _ = box_iou(first_anno, pred_box_s1)
                print(seq, 's1: ', iou.item())
                iou_all+=iou

                z_patch_arr, resize_factor_z, z_amask_arr = sample_target(image, init_box_s1.tolist()[0],
                                                                        self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                x_patch_arr, resize_factor_x, x_amask_arr = sample_target(image, init_box_s1.tolist()[0],
                                                                        self.params.search_factor,
                                                                        output_sz=self.params.search_size)
                template = self.preprocessor.process(z_patch_arr, z_amask_arr)
                sr = self.preprocessor.process(x_patch_arr, x_amask_arr)
                pr_s2 = self.prompt_extract_stage2(init_box_s1[0], resize_factor_x)
                box_ex_256_beforePrompt = torch.from_numpy(np.stack([x.numpy() for x in [pr_s2]])[None, :, :])
                prompt_id = self.refine_prompt(box_ex_256_beforePrompt.clone().detach().cpu())

                temp_list = [template.tensors] + [template.tensors.clone().detach() for _ in range(2)]

                box_mask_z, ce_keep_rate = self.ce(3)

                out_dict_2s = self.network.stage2(temp=temp_list,  # .module
                                                  search=[sr.tensors],
                                                  prompt=prompt_id[0],
                                                  s=2,
                                                  ce_template_mask=box_mask_z,
                                                  ce_keep_rate=ce_keep_rate,
                                                  return_last_attn=False,
                                                  )
                pred_score_map = out_dict_2s[-1]['score_map']
                response = self.output_window * pred_score_map  # win 0.5

                pred_boxes = self.network.box_head.cal_bbox(response, out_dict_2s[-1]['size_map'],
                                                            out_dict_2s[-1]['offset_map'], self.feat_sz)
                pred_boxes = pred_boxes[0].view(-1, 4)
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / resize_factor_x).tolist()  # (cx, cy, w, h) [0,1]
                # get the final box result
                self.state = init_box_s1[0]
                final_bbox = self.map_box_back(pred_box, resize_factor_x)
                fi = torch.tensor([[final_bbox[0].item(), final_bbox[1].item(), final_bbox[0].item()+final_bbox[2], final_bbox[1].item()+final_bbox[3]]])
                iou_2, _ = box_iou(first_anno, fi.cuda())
                iou_all_2 += iou_2
                print(seq, 's2: ', iou_2.item())

        print('Avg.IOU for stage1: ', iou_all.item()/len(test_seqs))
        print('Avg.IOU for stage2: ', iou_all_2.item()/len(test_seqs))


    def ce(self, n):
        box_mask_z = []
        ce_keep_rate = None
        if n==1:
            for i in range(n):
                box_mask_z.append(generate_point_cond(1, torch.device('cuda')))
        else:
            for i in range(n):
                box_mask_z.append(generate_mask_cond(self.cfg, 1, torch.device('cuda'), 0))
        box_mask_z = torch.cat(box_mask_z, dim=1)
        ce_start_epoch = 20
        ce_warm_epoch = 80
        ce_keep_rate = adjust_keep_rate(1000, warmup_epochs=ce_start_epoch,
                                        total_epochs=ce_start_epoch + ce_warm_epoch,
                                        ITERS_PER_EPOCH=1,
                                        base_keep_rate=0.7)
        return box_mask_z, ce_keep_rate

    def refine_prompt(self, anno): 
        N, B, _ = anno.shape
        anno = anno.unsqueeze(-2).repeat(1, 1, 25, 1).numpy()
        anno = torch.from_numpy(np.stack([anno[:, :, :, 0:2], anno[:, :, :, 2:4]], axis=2)).cuda()
        return anno.reshape(N, B, -1, 2)[:, :, :49, :].contiguous().float()

    def prompt_extract_stage2(self, box_extract: torch.Tensor, resize_factor: float) -> torch.Tensor:
        search_region_size = torch.Tensor([self.params.search_size, self.params.search_size])
        search_region_center = (search_region_size - 1) / 2
        scaled_box_wh = box_extract[2:4] * resize_factor
        box_in_search_region = torch.cat((search_region_center, scaled_box_wh))
        return box_in_search_region / search_region_size[0]

    def mapback(self, b2, map_param):
        for k in range(len(map_param['pad'])):
            if map_param['pad'][k] == 'x':
                b2[:, 0] -= 1
            else:
                b2[:, 1] -= 1
        b2 = b2 / map_param['fac']
        x, y, w, h = b2[0].clone().detach()
        b2[:, 2] = b2[:, 0] + 0.5*w
        b2[:, 3] = b2[:, 1] + 0.5*h
        b2[:, 0] = b2[:, 0] - 0.5* w
        b2[:, 1] = b2[:, 1] - 0.5* h

        return b2


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_tracker_class():
    return CATod



def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='lasot',
                    help='Name of dataset(lasot, lasot_extension_subset, trackingnet, got10k, uav123, uav112, dtb70).')

    args = parser.parse_args()

    param_module = importlib.import_module('lib.test.parameter.{}'.format('CATod'))
    params = param_module.parameters(args.tracker_param)
    tracker_module = importlib.import_module('lib.test.tracker.{}'.format('init_with_CATod'))
    tracker_class = tracker_module.get_tracker_class()
    tracker = tracker_class(params)
    tracker.initialize(args.dataset_name)

if __name__ == '__main__':
    main()