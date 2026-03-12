
import os
import torch.nn.functional as F
import torch
from torch import nn

from .head import build_box_head
from lib.models.CATos.vit_adp import vit_base_patch16_224
from timm.models.layers import trunc_normal_


class point_encoder(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, 768)
        self.norm = nn.LayerNorm(768)
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)
        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = self.norm(x + xs)
        else:
            x = self.norm(xs)
        return x

class CATos(nn.Module):
    """ This is the base class for CATos """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.point_adp = point_encoder(2, mlp_ratio=10, skip_connect=False)

        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, search: torch.Tensor,
                pro_id=None,
                tracking_st=False,
                ):

        out_dict = []
        for i in range(len(search)):

            embeds = self.point_adp(pro_id)  # B, 1, 768
            x, aux_dict = self.backbone(x=search[i], prompt=embeds)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            self.tracking_st = tracking_st

            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out_dict.append(out)

        return out_dict

    def stage2(self, temp=None, search=None, prompt=None, tracking_st=False, mode=None):

        out_dict = []
        if not prompt==None:
            embeds = self.point_adp(prompt)  # B, 1, 768
        else:
            embeds = [None]
        for i in range(len(search)):
            x, aux_dict = self.backbone.stage2(x=search[i], z=temp[i], prompt=embeds[0], mode=mode)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            self.tracking_st = tracking_st

            out = self.forward_head(feat_last, None, s='2')

            out.update(aux_dict)

            out_dict.append(out)

        return out_dict

    def forward_head(self, cat_feature, gt_score_map=None, s='1'):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        feat_len_s = cat_feature.shape[1]
        feat_sz_s = int(feat_len_s ** 0.5)
        enc_opt = cat_feature[:, -feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, feat_sz_s, feat_sz_s)

        if self.head_type == "CENTER":
            # run the center head
            if s=='2':
                score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map, s='2', tracking_st=self.tracking_st)
            else:
                score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map, tracking_st=self.tracking_st)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_cat_os(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, cfg=cfg)
        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError

    backbone.adp_pos_embed(cfg=cfg)

    box_head = build_box_head(cfg, hidden_dim)

    model = CATos(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    return model
