import os
import torch
from torch import nn

from .head import build_box_head
from lib.models.CATod.vit_ce import vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
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

class CATod(nn.Module):
    """ This is the base class for CATod """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        
        # track query: save the history information of the previous frame
        # self.track_query = nn.Parameter(torch.load('track_query.pth.tar', map_location='cuda'))
        self.track_query = None
        self.token_len = token_len

        self.point_adp = point_encoder(2, mlp_ratio=10, skip_connect=False)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                pro_id=None,
                tracking_st=False,
                s=0
                ):

        out_dict = []
        for i in range(len(search)):

            embeds = self.point_adp(pro_id)  # B, 1, 768

            x, aux_dict = self.backbone(z=template.copy(), x=search[i], prompt=embeds,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        track_query=self.track_query,
                                        token_len=self.token_len,
                                        pro_id=pro_id,
                                        s=s)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)  # 2, 576, 1
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()

            self.tracking_st = tracking_st

            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out_dict.append(out)

        return out_dict

    def stage2(self, temp=None, search=None, ce_template_mask=None, ce_keep_rate=None,
                return_last_attn=False, prompt=None, tracking_st=False, mode=None,s=0):

        out_dict = []
        if not prompt==None:
            embeds = self.point_adp(prompt)  # B, 1, 768
        else:
            embeds = None
        for i in range(len(search)):
            x, aux_dict = self.backbone.stage2(x=search[i], z=temp,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        track_query=self.track_query,
                                        return_last_attn=return_last_attn, prompt_feat=embeds, mode=mode,s=s)

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)  # 2, 576, 1
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()

            self.tracking_st = tracking_st

            out = self.forward_head(opt, None)

            out.update(aux_dict)

            out_dict.append(out)

        return out_dict

    def track(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                tracking_st=True
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone.forward_track(template.copy(), search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query,
                                        token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()  # stop grad  (B, N, C)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            self.tracking_st = tracking_st
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x

            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map, tracking_st=self.tracking_st)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_cat_od(cfg, training=True):
    pretrained = ''
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = CATod(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )
    return model
