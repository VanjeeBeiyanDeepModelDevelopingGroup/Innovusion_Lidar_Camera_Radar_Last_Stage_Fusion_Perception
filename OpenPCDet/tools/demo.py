import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin',data_file_list=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index,multi_frame=False):
        if multi_frame:
            multi_frame_diclist=[]
            if index < 3:
                for i in range(4):
                    if self.ext == '.bin':
                        points = np.fromfile(self.sample_file_list[3-i], dtype=np.float32).reshape(-1, 4)
                    elif self.ext == '.npy':
                        points = np.load(self.sample_file_list[3-i])
                    else:
                        raise NotImplementedError
                    data_dict=self.multi_process_info(points,index)
                    multi_frame_diclist.append(data_dict)
            else:
                for i in range(4):
                    if self.ext == '.bin':
                        points = np.fromfile(self.sample_file_list[index-i], dtype=np.float32).reshape(-1, 4)
                    elif self.ext == '.npy':
                        points = np.load(self.sample_file_list[index-i])
                    else:
                        raise NotImplementedError
                    data_dict=self.multi_process_info(points,index)
                    multi_frame_diclist.append(data_dict)
            return multi_frame_diclist
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    def multi_process_info(self,points,index):
        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # 注意如果用多帧的配置，需要在dataset.py的collate_batch和Demodataset的__getitem__中开启多帧
    parser.add_argument('--cfg_file', type=str, 
                        default='/data/Radar_Data/pcdet_cfg/models/pointpillar_7class.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/data/Radar_Data/lidar_camera/lidar/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, 
            default='/data/Radar_Data/OpenPCDet/output/ckpt/checkpoint_epoch_300.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    root_path=Path(args.data_path)
    ext='.bin'
    import os
    test_list = os.listdir(root_path)
    num_test = len(test_list)
    train_list_queue=[]
    # for i in range(num_test):
    #     train_list =  '%d.bin'%i
    #     train_list_queue.append('/data/lianghao/3D_val_data/bin/'+train_list) 

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger#,data_file_list=train_list_queue
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) #list[]
            idx_num=str(idx)
            # import pdb;pdb.set_trace()
            # pred_box = pred_dicts[0]['pred_boxes'].detach().cpu()
            # pred_scores = pred_dicts[0]['pred_scores'].detach().cpu()
            # pred_labels = pred_dicts[0]['pred_labels'].detach().cpu()
            # pred_scores = torch.cat([torch.unsqueeze(pred_scores,1),torch.unsqueeze(pred_labels,1)],dim=1)
            # pred_result = torch.cat([pred_box,pred_scores],dim=1)
            # import pdb;pdb.set_trace()
            # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(pred_result[:, 7], descending=True)
            # pred_result_sort = pred_result[conf_sort_index]
            # import pdb;pdb.set_trace()
            # np.savetxt('/data/lianghao/pointpillar_result/pred_result_pointpillar_mf_300epoch/'+idx_num+'.csv', pred_result_sort, delimiter=",")
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], 
            #     ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], 
            #     ref_labels=pred_dicts[0]['pred_labels']
            # )
            # import pdb;pdb.set_trace()
            V.draw_scenes(
                points=data_dict['points'][:, 1:], 
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], 
                ref_labels=pred_dicts[0]['pred_labels']
            )
            # if idx ==630:
                # break
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
