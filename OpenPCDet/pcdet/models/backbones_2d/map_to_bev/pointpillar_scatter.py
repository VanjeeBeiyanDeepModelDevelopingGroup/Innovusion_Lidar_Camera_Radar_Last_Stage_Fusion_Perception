import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    对应到论文中就是stacked pillars,将生成的pillar按照坐标索引还原到原空间中
    在PillarVFE之后就是这个操作
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        Args: 
            pillar_features:(189500,64)
            coords:(189500, 4) 第一维是batch_index
        Returns:
            batch_spatial_features:(2, 64, 496, 496)
        """
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1 # 8
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # reshape回原空间(伪图像)--> (8, 64, 496, 496) 
        # batch [0,1,2,...,4*batch_size]
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_mf(nn.Module):
    """
    多帧
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        Args: 
            pillar_features:(189500,64)
            coords:(189500, 4) 第一维是batch_index
        Returns:
            batch_spatial_features:(2, 64, 496, 496)
        """
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1 # 8
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # reshape回原空间(伪图像)--> (8, 64, 496, 496) 
        # batch [0,1,2,...,4*batch_size]
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        # 如果是多帧 precat
        # import pdb;pdb.set_trace()
        if batch_size != batch_dict['batch_size']:
            multi_frame_batch=[]
            for i in range(batch_dict['batch_size']):
                # 连续4帧 当前帧为frame_t, frame_t1为t-1帧 [64, 496, 432]
                frame_t = batch_spatial_features[4*i].unsqueeze(0)
                frame_t1 = batch_spatial_features[4*i+1].unsqueeze(0)
                frame_t2 = batch_spatial_features[4*i+2].unsqueeze(0)
                frame_t3 = batch_spatial_features[4*i+3].unsqueeze(0)
                # precat
                pesu_img_t=attention_base(frame_t,frame_t,frame_t)
                pesu_img_t1=attention_base(frame_t,frame_t1,frame_t1)
                pesu_img_t2=attention_base(frame_t,frame_t2,frame_t2)
                pesu_img_t3=attention_base(frame_t,frame_t3,frame_t3)
                
                pesu_img_all = torch.cat([pesu_img_t,pesu_img_t1,pesu_img_t2,pesu_img_t3],dim=1) #([1, 256, 640, 640])
                multi_frame_batch.append(pesu_img_all.squeeze(0))
            # import pdb;pdb.set_trace()
            # [2,256,640,640]
            multi_frame_batch=torch.stack(multi_frame_batch,0)
            batch_dict['spatial_features'] = multi_frame_batch
        else:
            batch_dict['spatial_features'] = batch_spatial_features
        # 下一个模块：pcdet/models/backbones_2d/base_bev_backbone.py
        return batch_dict

def attention_base(q, k, v):
    # attention(Q,K,V)=softmax(Q*K.T/d)*V
    
    bs,c,w,h = q.shape
    weight_mat = torch.matmul(q,k.permute(0,1,3,2))
    weight_mat = weight_mat.reshape(bs,c,w*h) # [1, 64, 640, 640]
    weight_mat = torch.softmax(weight_mat,dim=2)
    weight_mat = weight_mat.reshape(bs, c, w, h)
    output = weight_mat.mul(v)+v
    return output