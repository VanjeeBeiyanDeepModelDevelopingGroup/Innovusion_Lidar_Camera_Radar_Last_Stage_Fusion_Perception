from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    #调用detectors中__init__.py中的build_detector构建网络
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    """
    跳过metadata,frame_id,calib,同时根据数据类型转换数据类型,再放到gpu上
    """
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    """
    模型函数装饰器
    """
    # 定义一个namedtuple类型: https://blog.csdn.net/kongxx/article/details/51553362
    # 包括平均损失，tensorboard结果字典，显示结果字典
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # 这里输入数据，得到结果字典
        # import pdb;pdb.set_trace()
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        # 计算平均损失
        loss = ret_dict['loss'].mean()
        # 更新迭代次数，Detector3DTemplate中进行注册global_step
        # global_step在滑动平均、优化器、指数衰减学习率等方面都有用到
        # global_step的初始化值是0
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()
        # 返回ModelReturn对象
        # 简单来说返回的ModelReturn对象实例化之后可以通过 x.loss获取loss值
        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
