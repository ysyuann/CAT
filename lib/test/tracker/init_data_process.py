import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import random
import cv2
import numpy as np
import sys
import random
import copy
import math
from torch import Tensor
from PIL import Image

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


def gener_point_newyys(gt):
    si = 320
    gt_i = gt.clone().detach()
    gt_i[2] = gt_i[2] + gt_i[0]
    gt_i[3] = gt_i[3] + gt_i[1]
    gt_i = gt_i.clamp(min=0.0, max=si-1)  # xyxy

    mode = random.choices(['scr', 'point_single', 'mohu_box'], weights=[1, 0, 0], k=1)[0]  #   gai1
    # mode = random.choices(['scr'], weights=[1], k=1)[0]
    if mode == 'scr':
        x1, y1, x2, y2 = gt_i[:]
        w = x2 - x1
        h = y2 - y1
        scale_bbox = jit(torch.tensor([x1, y1, w, h]), 0.5)
        jit_box = get_jittered_box(scale_bbox, 0.05, 0.3)
        gt_j = jit_box
        gt_j[2] = gt_j[2] + gt_j[0]
        gt_j[3] = gt_j[3] + gt_j[1]
        gt_j = gt_j.clamp(min=0.0, max=si - 1)  # xyxy
        mask = torch.zeros([1, 1, int(si), int(si)], dtype=int)
        mask[0, 0, int(gt_j[1]):int(gt_j[3] + 1), int(gt_j[0]):int(gt_j[2] + 1)] = 1
        gt_j = gt_j.unsqueeze(0).unsqueeze(0)
        b = forward_point(mask, gt_j)
        p = random.choices([1, 2], weights=[1, 0], k=1)[0]
        if p == 1:
            if b['rand_shape'].sum()!=0:
                b1 = point(b['rand_shape'].view(int(si), int(si)))  # need to be 2d
                b1 = {'rand_shape': b1}  # need to be 3d
            else:
                b1 = b
        else:
            b1 = b

        if b1['rand_shape'].sum()!=0:
            return samp(b1['rand_shape'][0]), mask
        else:
            return 0
        # return 1


    elif mode == 'point_single':
        x1, y1, x2, y2 = gt_i[:]
        w = int(x2 - x1)
        h = int(y2 - y1)
        cx = int(x1 + 0.5 * w)
        cy = int(y1 + 0.5 * h)
        new_w = int(w / 10)
        new_h = int(h / 10)
        mask = torch.zeros([1, int(si), int(si)], dtype=int)
        if new_w * new_h > 25:
            s_w = int(w / 20)
            s_h = int(h / 20)
            j_w = random.randint(-s_w, s_w)
            j_h = random.randint(-s_h, s_h)
        else:
            j_w = random.randint(-1, 1)
            j_h = random.randint(-1, 1)
        # mask[0, (cy + j_h-1):(cy + j_h+1), (cx + j_w-1):(cx + j_w+1)] = 1    # gai2
        mask[0, cy, cx] = 1  # gai2
        mask = mask == 1

        if mask.sum()!=0:
            return samp(mask[0]), mask
        else:
            return 0
        # return 1

    elif mode == 'mohu_box':
        x1, y1, x2, y2 = gt_i[:]
        w = x2 - x1
        h = y2 - y1
        p = random.choices([1, 2], weights=[3, 1], k=1)[0]
        mask = torch.zeros([1, int(si), int(si)], dtype=int)
        p=2
        if p == 1:  # rand_point
            jit_box = get_jittered_box(torch.tensor([x1, y1, w, h]), 0.15, 0.4)  # xywh 0.05, 0.3
            x1, y1, w, h = jit_box[:]
            x2 = x1 + w
            y2 = y1 + h
            mask[0, int(y1):int(y2 + 1), int(x1):int(x2 + 1)] = 1

        else:  # 2_point
            jit_box = get_jittered_box(torch.tensor([x1, y1, w, h]), 0.15, 0.4)  # 0.05, 0.1
            x1, y1, w, h = jit_box[:]
            x2 = x1 + w
            y2 = y1 + h
            mask[0, int(y1), int(x1)] = 1                        # gai3
            mask[0, int(y2), int(x2)] = 1
            # mask[0, int(y1-1):int(y1+1), int(x1-1):int(x1+1)] = 1
            # mask[0, int(y2-1):int(y2+1), int(x2-1):int(x2+1)] = 1

        mask = mask == 1

        if mask.sum()!=0:
            return samp(mask[0]), mask
        else:
            return 0
        # return 1


def samp(mask):
    wh = mask.shape[-1]

    pm = mask
    pm_nz = pm.nonzero()
    length = pm_nz.shape[0]
    if length >= 30:
        idx = torch.randperm(length)[:30].tolist()
    else:
        # n = 30 / length
        p1 = [j for j in range(length)]
        idx = (p1 * 30)[:30]
    idx.sort()

    return pm_nz[idx] / wh


def click_sample(gt, si, out_sz, factor=20):
    hs, ws = si
    gt_i = gt.clone()

    # Convert from [x,y,w,h] to [x1,y1,x2,y2]
    gt_i[2:] += gt_i[:2]

    # Clamp all coordinates to image bounds [0, width-1] or [0, height-1]
    gt_i.clamp_(min=0.0)
    gt_i[0::2].clamp_(max=ws - 1)  # Clamp x-coords (x1, x2)
    gt_i[1::2].clamp_(max=hs - 1)  # Clamp y-coords (y1, y2)

    x1, y1, x2, y2 = gt_i[:]
    w = int(x2 - x1)
    h = int(y2 - y1)
    cx = int(x1 + 0.5 * w)
    cy = int(y1 + 0.5 * h)

    fac = 10
    thred_w = int(w / fac)
    thred_h = int(h / fac)
    mask = torch.zeros([1, int(hs), int(ws)], dtype=int)
    if thred_w * thred_h > 25:
        s_w = int(w / factor)  # 20
        s_h = int(h / factor)
        j_w = random.randint(-s_w, s_w)
        j_h = random.randint(-s_h, s_h)
    else:
        j_w = random.randint(-1, 1)
        j_h = random.randint(-1, 1)

    mask[0, cy + j_h, cx + j_w] = 1
    mask = mask == 1

    # if mask.sum()!=0:
    num = 30 if out_sz == 320 else 46
    pm_nz = mask[0].nonzero()
    idx = [0] * num
    return pm_nz[idx], mask
    # else:
    #     return 0


def samp1(mask, out_sz):
    wh = mask.shape[-1]
    num = 30 if out_sz==320 else 46 # 46 for 384, 30 for 256
    pm = mask
    pm_nz = pm.nonzero()
    length = pm_nz.shape[0]
    if length >= num:
        idx = torch.randperm(length)[:num].tolist()
    else:
        # n = 30 / length
        p1 = [j for j in range(length)]
        idx = (p1 * num)[:num]
    idx.sort()

    return pm_nz[idx] #/ wh


def point(mask):
    b = mask.nonzero()
    min_point_x = int(b[:, 0].min())
    min_point_y = int(b[:, 1].min())
    max_point_x = int(b[:, 0].max())
    max_point_y = int(b[:, 1].max())
    mask[min_point_x: max_point_x+1, min_point_y: max_point_y+1] = 1
    return mask.unsqueeze(0)

def get_jittered_box(box, s, c):
    """ Jitter the input box
    args:
        box - input bounding box
        mode - string 'template' or 'search' indicating template or search data

    returns:
        torch.Tensor - jittered box
    """

    jittered_size = box[2:4] * torch.exp(torch.randn(2) * s)
    max_offset = (jittered_size.prod().sqrt() * torch.tensor(c).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

def jit(bbox, s):
    x, y, w, h = bbox[:]
    # scale = 1
    x1 = x - s * w * 0.5 + w * 0.5
    y1 = y - s * h * 0.5 + h * 0.5
    return torch.tensor([x1, y1, s * w, s * h])

def gener_point(gt):
    si = 320
    gt_i = gt.clone().detach()
    gt_i[2] = gt_i[2] + gt_i[0]
    gt_i[3] = gt_i[3] + gt_i[1]
    gt_i = gt_i.clamp(min=0.0, max=si-1)

    mask = torch.zeros([1, 1, int(si), int(si)], dtype=int)
    mask[0, 0, int(gt_i[1]):int(gt_i[3]+1), int(gt_i[0]):int(gt_i[2]+1)] = 1
    gt_i = gt_i.unsqueeze(0).unsqueeze(0)

    b = forward_point(mask, gt_i)
    # gt = np.array(mask, bbox)
    # b = np.array(b['rand_shape']).reshape(int(si), int(si))
    # # if b.sum()==0:
    # #     aa=1
    # im = PIL.Image.fromarray(b)
    # name = str(random.randint(1, 200))
    # im.convert('L').save('/15716608638/tu/'+name+'.bmp')

    # a = 1
    return b

def forward_point(masks, boxes):
    # masks = instances.gt_masks.tensor
    # boxes = instances.gt_boxes.tensor

    indices = [x for x in range(len(masks))]

    random.shuffle(indices)
    candidate_mask = masks[indices[0]]
    candidate_box = boxes[indices[0]]

    shape_candidate = ['Circle', 'Point', 'Polygon', 'Scribble' ]  # 'Circle', 'Point', 'Polygon', 'Scribble'
    shape_prob = [1, 1, 1, 1]  # , 1, 1, 1
    cans = [getattr(sys.modules[__name__], class_name)(0, True) for class_name in shape_candidate]
    draw_funcs = random.choices(cans, weights=shape_prob, k=len(candidate_mask))

    rand_shapes = [d.draw(x, y) for d, x, y in zip(draw_funcs, candidate_mask, candidate_box)]

    # x1, y1, x2, y2 = int(candidate_box[0, 1]), int(candidate_box[0, 0]), int(candidate_box[0, 3]), int(candidate_box[0, 2])
    # rand_shapes[0][x1-1:x1+1, y1-1:y1+1] = True
    # rand_shapes[0][x2-1:x2+1, y2-1:y2+1] = True

    types = [repr(x) for x in draw_funcs]
    for i in range(0, len(rand_shapes)):
        if rand_shapes[i].sum() == 0:
            candidate_mask[i] = candidate_mask[i] * 0
            # types[i] = 'none'

    # candidate_mask: (c,h,w), bool. rand_shape: (c, iter, h, w), bool. types: list(c)
    return {'rand_shape': torch.stack(rand_shapes).bool()}

def grounding_resize(im, output_sz, bbox, mask=None):
    """ Resize the grounding image without change the aspect ratio, First choose the short side,then resize_factor =
    scale_factor * short side / long size, then padding the border with value 0

    args:
        im - cv image
        output_sz - return size of img int
        bbox - the bounding box of target in image , which form is (X, Y, W, H)
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mask - the image of mask which size is [H, W] numpy array
    returns:
        im_crop_padded  - resized and padded image which shape is (resize_H, resize_W, C)
        box - resize and normalize, the coord is normalized to [0,1]
        att_mask - shape is (resize_H, resize_W)  the value of padding pixel is 1, the original pixel is 0
        mask_crop_padded - all zero and shape is (H, W)
    """
    # resize img
    h, w = im.shape[0:-1]
    # scale_factor = random.uniform(0.5, 1)
    scale_factor = 1
    crop_sz = math.ceil(scale_factor * output_sz)
    interpolation = Image.BILINEAR
    if w > h:
        ow = crop_sz
        oh = int(crop_sz * h / w)
    else:
        oh = crop_sz
        ow = int(crop_sz * w / h)
    # resie image
    img = cv2.resize(im, (ow, oh), interpolation)

    new_h, new_w = img.shape[0:2]
    # print(f'new_w,new_h = {new_w},{new_h}')
    # 居中 Padding
    # y1_pad = int((output_sz - new_h) / 2)
    # y2_pad = int((output_sz - new_h) / 2)
    # x1_pad = int((output_sz - new_w) / 2)
    # x2_pad = int((output_sz - new_w) / 2)
    # 只Padding下面
    y1_pad = 0
    y2_pad = int((output_sz - new_h))
    x1_pad = 0
    x2_pad = int((output_sz - new_w))
    if (y1_pad + y2_pad + new_h) != output_sz:
        y1_pad += 1
    if (x1_pad + x2_pad + new_w) != output_sz:
        x1_pad += 1
    box = copy.deepcopy(bbox)

    # scale the box size
    box[0] = bbox[0] * new_w / w
    box[1] = bbox[1] * new_h / h
    box[2] = bbox[2] * new_w / w
    box[3] = bbox[3] * new_h / h

    assert (y1_pad + y2_pad + new_h) == output_sz and (x1_pad + x2_pad + new_w) == output_sz, print(
        'y1_pad:{},y2_pad:{},x1_pad:{},x2_pad:{}'.format(y1_pad, y2_pad, x1_pad, x2_pad)) and print(
        f'img shape:{img.shape}')
    # the left top coord of the resized image in the padding image
    image_top_coords = [x1_pad, y1_pad, new_w, new_h]
    # Pad
    im_crop_padded = cv2.copyMakeBorder(img, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, (0, 0, 0))
    # add the padding distance
    box[0] += x1_pad
    box[1] += y1_pad
    # normalized to [0,1]
    box /= output_sz

    H, W, _ = im_crop_padded.shape
    if mask is not None:
        # todo find a better way to resize mask, mask is a tensor which all values is zero
        mask_crop_padded = torch.zeros(H, W)
        # mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
    else:
        mask_crop_padded = torch.zeros(H, W)

    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    return (im_crop_padded, ), [box], (att_mask, ), (mask_crop_padded, )  # , image_top_coords


def grounding_resize1(im, output_sz, bbox, mask=None, pr_points = None):
    """ Resize the grounding image without change the aspect ratio, First choose the short side,then resize_factor =
    scale_factor * short side / long size, then padding the border with value 0

    args:
        im - cv image
        output_sz - return size of img int
        bbox - the bounding box of target in image , which form is (X, Y, W, H)
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mask - the image of mask which size is [H, W] numpy array
    returns:
        im_crop_padded  - resized and padded image which shape is (resize_H, resize_W, C)
        box - resize and normalize, the coord is normalized to [0,1]
        att_mask - shape is (resize_H, resize_W)  the value of padding pixel is 1, the original pixel is 0
        mask_crop_padded - all zero and shape is (H, W)
    """
    # resize img
    h, w = im.shape[0:-1]
    # scale_factor = random.uniform(0.5, 1)
    scale_factor = 1
    crop_sz = math.ceil(scale_factor * output_sz)
    interpolation = Image.BILINEAR
    if w > h:
        ow = crop_sz
        oh = int(crop_sz * h / w)

    else:
        oh = crop_sz
        ow = int(crop_sz * w / h)

    # resie image
    img = cv2.resize(im, (ow, oh), interpolation)

    new_h, new_w = img.shape[0:2]
    # print(f'new_w,new_h = {new_w},{new_h}')
    # 居中 Padding
    # y1_pad = int((output_sz - new_h) / 2)
    # y2_pad = int((output_sz - new_h) / 2)
    # x1_pad = int((output_sz - new_w) / 2)
    # x2_pad = int((output_sz - new_w) / 2)
    # 只Padding下面
    y1_pad = 0
    y2_pad = int((output_sz - new_h))
    x1_pad = 0
    x2_pad = int((output_sz - new_w))
    if (y1_pad + y2_pad + new_h) != output_sz:
        y1_pad += 1
    if (x1_pad + x2_pad + new_w) != output_sz:
        x1_pad += 1
    box = copy.deepcopy(bbox)

    # scale the box size
    box[0] = bbox[0] * new_w / w
    box[1] = bbox[1] * new_h / h
    box[2] = bbox[2] * new_w / w
    box[3] = bbox[3] * new_h / h
    pr_points[:, 0] = pr_points[:, 0] * new_h / h
    pr_points[:, 1] = pr_points[:, 1] * new_w / w
    pr_points[:, 0] += y1_pad
    pr_points[:, 1] += x1_pad

    assert (y1_pad + y2_pad + new_h) == output_sz and (x1_pad + x2_pad + new_w) == output_sz, print(
        'y1_pad:{},y2_pad:{},x1_pad:{},x2_pad:{}'.format(y1_pad, y2_pad, x1_pad, x2_pad)) and print(
        f'img shape:{img.shape}')
    # the left top coord of the resized image in the padding image
    image_top_coords = [x1_pad, y1_pad, new_w, new_h]
    # Pad
    im_crop_padded = cv2.copyMakeBorder(img, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, (0, 0, 0))
    # add the padding distance
    box[0] += x1_pad
    box[1] += y1_pad
    # normalized to [0,1]
    box /= output_sz

    H, W, _ = im_crop_padded.shape
    if mask is not None:
        # todo find a better way to resize mask, mask is a tensor which all values is zero
        mask_crop_padded = torch.zeros(H, W)
        # mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
    else:
        mask_crop_padded = torch.zeros(H, W)

    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    map_param = {}
    map_param['fac'] = new_w / w
    map_param['pad'] = []
    if x1_pad != 0:
        map_param['pad'].append('x')
    if y1_pad != 0:
        map_param['pad'].append('y')

    return (im_crop_padded, ), [box], (att_mask, ), (mask_crop_padded, ), pr_points/float(output_sz), map_param  # , image_top_coords



def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union










