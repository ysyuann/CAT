import numpy as np
from lib.models.CATod import build_cat_od
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.test.tracker.test_dataset_info import get_seq_pth, init_resize, get_test_seq_list
from lib.test.tracker.init_data_process import click_sample, box_iou
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate, generate_point_cond

class CATod(BaseTracker):
    def __init__(self, params):
        super(CATod, self).__init__(params)
        network = build_cat_od(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        network.track_query = torch.load('/home/space/works/ODTrack/tracking/track_query_for_init.pth.tar',
                                         map_location='cuda')

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.iou_all_2 = torch.tensor([0], dtype=torch.float32).cuda()  # cal_IOU
        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict, load_init=0, init_select=None):

        first_anno = info['init_bbox']

        h, w = image.shape[0:-1]
        mask_points_ori, _ = click_sample(torch.tensor(first_anno), [h, w], self.params.search_size,
                                          prompt_len=49)  # tensor 30, 2

        x_patch_arr, x_bbox, x_amask_arr, mask_pad, mask_point, map_param = init_resize(image,
                                                                                        int(self.params.search_size),
                                                                                        torch.tensor(first_anno),
                                                                                        pr_points=mask_points_ori)
        x_patch_arr1 = self.preprocessor.process(x_patch_arr[0], x_amask_arr[0])
        x = x_patch_arr1.tensors

        prompt_id = mask_point.unsqueeze(0).cuda()
        with torch.no_grad():
            box_mask_z, ce_keep_rate = self.ce(1)

            out_dict = self.network(template=[None], search=[x], pro_id=prompt_id, s=1,
                                    ce_template_mask=box_mask_z,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=False, )
            pred_box_s1 = out_dict[0]['pred_boxes'][0] * self.params.search_size

            first_anno = torch.tensor(first_anno).cuda().reshape(1, 4)
            first_anno[:, 2:] += first_anno[:, :2]  # (x, y, w, h) -> (x1, y1, x2, y2)

            pred_box_s1 = self.mapback(pred_box_s1, map_param)

            init_box_s1 = pred_box_s1.clone().detach().cpu()
            init_box_s1[:, 2:] -= init_box_s1[:, :2]

            iou, _ = box_iou(first_anno, pred_box_s1)
            print('s1: ', iou.item())

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
            init_box_s2 = torch.tensor([[final_bbox[0].item(), final_bbox[1].item(), final_bbox[0].item() + final_bbox[2],
                                final_bbox[1].item() + final_bbox[3]]])
            iou_2, _ = box_iou(first_anno, init_box_s2.cuda())
            info['init_bbox'] = [final_bbox[0].item(), final_bbox[1].item(), final_bbox[2], final_bbox[3]]
            print('s2: ', iou_2.item())
            # self.iou_all_2 += iou_2
        # print(self.iou_all_2/70)
        self.network.track_query = None
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # --------- select memory frames ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            out_dict = self.network.track(template=template_list, search=[search.tensors], ce_template_mask=box_mask_z)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # --------- save memory frames and masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()
        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save memory frames and masks ---------
        
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None

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

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return CATod
