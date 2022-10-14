#***需要修改label_name与class_name
from audioop import mul
import copy
import os
import pdb
import pickle

import numpy as np
from skimage import io
import os
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import object3d_wj
from pcdet.datasets.dataset import DatasetTemplate

# 定义自己的数据集
# 这里继承了DatasetTemplate类
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        # 确认mode=train or val
        # import pdb;pdb.set_trace()
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('single-infrastructure-side-example' if self.split != 'val' else 'single-infrastructure-side-example')

        self.sample_id_list = [x for x in range(len(os.listdir(str(self.root_split_path)+'/bin/')))] 
        self.myself_infos = []
        # 加载myself数据
        self.include_myself_data(self.mode)

    def include_myself_data(self, mode):
        if self.logger is not None:
            # 如果日志信息存在，则加入'Loading KITTI dataset'的信息
            self.logger.info('Loading KITTI dataset')
        # 创建新列表，用于存放信息
        myself_infos = []

        '''   
        INFO_PATH: {
        'train': [kitti_infos_train.pkl],
        'test': [kitti_infos_val.pkl],}
        '''
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            # root_path = '/data/lianghao/'
            info_path = self.root_path / info_path
            # info_path = '/data/lianghao/kitti_infos_train.pkl' or '/data/lianghao/kitti_infos_val.pkl'
            if not info_path.exists():
                # 如果该文件不存在，跳出，继续下一个文件。
                continue
            # 打开该文件
            with open(info_path, 'rb') as f:
                # 打开该pkl文件，并将解析内容添加到myself_infos列表中
                infos = pickle.load(f)
                myself_infos.extend(infos)

        self.myself_infos.extend(myself_infos)

        if self.logger is not None:
            self.logger.info('Total samples for myself dataset: %d' % (len(myself_infos)))

    def set_split(self, split):
        """ get tag list according to the split

        Args:
            split(string): train or test

        Returns:
            list: list of tag

        """
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('single-infrastructure-side-example' if self.split != 'val' else 'single-infrastructure-side-example')

        self.sample_id_list = [x for x in range(len(os.listdir(str(self.root_split_path)+'/bin/')))] 

    def get_lidar(self, idx):
        # 就是读点云的bin
        lidar_file = self.root_split_path / 'bin' / ('%06d.bin' % idx)
        assert lidar_file.exists()

        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        # 获取label信息
        label_file = self.root_split_path / 'label'/'virtuallidar' / ('%06d.json' % idx)

        print(label_file)
        # import pdb;pdb.set_trace()
        assert label_file.exists()
        # 调用get_objects_from_label函数，首先读取该文件的所有行赋值为 lines
        # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
        # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等
        return object3d_wj.get_objects_from_label(label_file)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 定义了一个info字典
            info = {}
            # 点云信息:特征数和点的索引
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            # 添加点云信息到info里
            info['point_cloud'] = pc_info
            label_path = '/data/Radar_Data/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side/label/virtuallidar'
            if has_label:
                # 根据索引读取label
                label_list = os.listdir(label_path)
                label_list.sort()

                # import pdb;pdb.set_trace()
                # obj_list = self.get_label(sample_idx)
                obj_list = object3d_wj.get_objects_from_label(label_path+'/'+label_list[sample_idx])
                annotations = {}
                # 根据label将所有属性添加到annotations
                # 类别名name，角度alpha，长高宽dimensions，中心坐标xyz location
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list]) 
                annotations['location'] = np.array([obj.loc for obj in obj_list])
                # 计算目标数量，除去DontCare，咱们的数据好像是没有这个东西。
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                # 如果是DontCare就置-1
                # example 5个样本，2个DontCare index = [0,1,2,-1,-1]
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                # 由于DontCare是放在最后的，所以可以[:num_objects]取有效目标
                yaw = annotations['alpha'][:num_objects]
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]

                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                yaw = np.array(yaw).reshape((-1, 1))

                gt_boxes_lidar = np.concatenate([loc, l, w, h, yaw], axis=1)
                # 把annotations放到info里 key='annos'
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # 多线程异步处理，反正就是加速的
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # 返回一个列表infos，每一个元素是一个字典，代表一帧的信息
        return list(infos)


    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='training'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'training' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('myself_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        bin_path = '/data/Radar_Data/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-example/bin'
        bin_list = os.listdir(bin_path)
        bin_list.sort()
        # 配置里写的ROOT_DIR + kitti_infos_train.pkl
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 读取列表infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            # 输出的是第几个样本，如 2/500
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            # 取当前帧信息 info
            info = infos[k]
            # info['point_cloud']['lidar_idx']是点云索引
            # 在我们的数据集里其实等同于frame id
            sample_idx = info['point_cloud']['lidar_idx']
            # bin转numpy格式，points.shape=(M,4)
            points = np.fromfile(bin_path +'/'+bin_list[sample_idx], dtype=np.float32).reshape(-1, 4)
            # points = self.get_lidar(sample_idx)
            # 读info里的annos，是label的一些信息。
            # 类别名name，角度alpha，长高宽dimensions，中心坐标xyz location
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']
            # 有效目标个数
            num_obj = gt_boxes.shape[0]
            # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            # 遍历每个目标框
            for i in range(num_obj):
                # 创建文件名，如 0_car_1.bin
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                # /data/lianghao/gt_database/0_car_1.bin
                filepath = database_save_path / filename
                # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                # 再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                # 用点云的点减框的中心点坐标
                gt_points[:, :3] -= gt_boxes[i, :3]
                # 把gt_points写入文件
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # 获取文件相对路径
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # 根据当前物体的信息组成info
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]
                                }
                    # 把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        # 如果存在该类别则追加
                        all_db_infos[names[i]].append(db_info)
                    else:
                        # 如果不存在该类别则新增
                        all_db_infos[names[i]] = [db_info]
        # 输出数据集中不同类别物体的个数                
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
        # 把所有的all_db_infos写入到文件里面
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'alpha': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict


            box_num = pred_boxes.shape[0]
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['alpha'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            # import pdb;pdb.set_trace()
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(loc)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['alpha'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def convert_detection_to_wjdata_annos(self, detections):
        annos = []
        # label_name=['car', 'bicycle', 'bus', 'van', 'pedestrian', 'semitrailer', 'truck']
        #label_name=['car', 'bicycle', 'bus',  'tricycle', 'pedestrian','semitrailer', 'truck', 'jtruck']
        label_name=['Bus', 'Car', 'Truck', 'Van','Pedestrian','Cyclist','Trafficcone','Motorcyclist','Tricyclist','Barrowlist']
        for detection in detections:
            anno = {}
            dt_boxes = detection["box3d_lidar"].detach().cpu().numpy()
            box_num = dt_boxes.shape[0]
            labels = detection["label_preds"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            anno["score"] = scores
            anno["dimensions"] = dt_boxes[:, 3:6]
            anno["location"] = dt_boxes[:, :3]
            anno["rotation_y"] = dt_boxes[:, -1]
            anno["name"] = [label_name[int(label)] for label in labels]
            # anno["gt_labels"] = np.array([self._cls2label[cls] for cls in anno["name"]])
            annos.append(anno)

        return annos

    def multi_process_info(self,info):
        # 获取采样的序列号 
        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            # 获取该帧信息的 'annos'
            annos = info['annos']
            # gt_names 返回当前帧所有目标的类别
            # 例如：gt_names = ['bicycle', 'pedestrian', 'car']
            gt_names = annos['name']
            # gt_boxes_liadar返回的是gt_names中每类物体的box[N,7]
            gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points
        # 将输入数据送入prepare_data进一步处理，形成训练数据
        # 这里包含了数据增强步骤
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.myself_infos[0].keys():
            # 如果'annos'没在kitti信息里面，直接返回空字典，如果has_label=True则会构建annos
            return None, {}

        from .kitti_object_eval_python import eval as wj_eval
        # import pdb;pdb.set_trace()
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.myself_infos]
        # ap_result_str, ap_dict = wj_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        result_official_dict = wj_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        # return ap_result_str, ap_dict
        return result_official_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.myself_infos) * self.total_epochs

        return len(self.myself_infos)

    def __getitem__(self, index, multi_frame = False,mode= 'train'):
        """
        从pkl文件中获取相应index的info,然后根据info['point_cloud']['lidar_idx']确定帧号,进行数据读取和其他info字段的读取
        初步读取的data_dict,要传入prepare_data(dataset.py父类中定义)进行统一处理，然后即可返回
        """
        # multi frame mode
        multi_frame_diclist=[]
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.myself_infos)
        # 这里开始取多帧信息，如果帧号小于3则取0-3帧
        if multi_frame:
            if mode == 'train':
                if index < 3:
                    for i in range(4):
                        # [3,2,1,0]
                        info = copy.deepcopy(self.myself_infos[3-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                # 如果帧号在2400至2403则取2400 2401 2402 2403帧
                elif index in range(2400,2404): # 2400 2401 2402 2403
                    for i in range(4):
                        # [2403,2402,2401,2400]
                        info = copy.deepcopy(self.myself_infos[2403-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                elif index in range(5400,5404): # 5400 5401 5402 5403
                    for i in range(4):
                        # [5403,5402,5401,5400]
                        info = copy.deepcopy(self.myself_infos[5403-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                else:
                    for i in range(4):
                        # [index,index-1,index-2,index-3]
                        info = copy.deepcopy(self.myself_infos[index-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                # import pdb;pdb.set_trace() 
                return multi_frame_diclist
            if mode == 'val':
                if index < 3:
                    for i in range(4):
                        # [3,2,1,0]
                        info = copy.deepcopy(self.myself_infos[3-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                elif index in range(632,636): # 632 633 634 635
                    for i in range(4):
                        # [2403,2402,2401,2400]
                        info = copy.deepcopy(self.myself_infos[635-i])
                        data_dict=self.multi_process_info(info)
                        multi_frame_diclist.append(data_dict)
                return multi_frame_diclist
        # 取出第index帧的信息
        # 可以从这里改多帧
        # self.myself_infos是List 每个元素是一个字典，index等同于帧号
        info = copy.deepcopy(self.myself_infos[index])
        

        
        # 获取采样的序列号 
        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            # 获取该帧信息的 'annos'
            annos = info['annos']
            # gt_names 返回当前帧所有目标的类别
            # 例如：gt_names = ['bicycle', 'pedestrian', 'car']
            gt_names = annos['name']
            # gt_boxes_liadar返回的是gt_names中每类物体的box[N,7]
            gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        if "points" in get_item_list:
            bin_path = '/data/Radar_Data/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-example/bin'
            bin_list = os.listdir(bin_path)
            bin_list.sort()
            # import pdb;pdb.set_trace()
            # points = self.get_lidar(sample_idx)
            points = np.fromfile(bin_path+'/'+bin_list[sample_idx], dtype=np.float32).reshape(-1, 4)
            input_dict['points'] = points
        # 将输入数据送入prepare_data进一步处理，形成训练数据
        # 这里包含了数据增强步骤
        data_dict = self.prepare_data(data_dict=input_dict)
        # import pdb;pdb.set_trace()
        # data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('myself_infos_%s.pkl' % train_split)
    val_filename = save_path / ('myself_infos_%s.pkl' % val_split)
    # trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    # test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info training file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)
    # import pdb;pdb.set_trace()
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':

    # if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':

        import sys
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open("/data/Radar_Data/pcdet_cfg/dataset/wanji_dataset_7class.yaml")))
        ROOT_DIR = (Path(__file__).resolve().parent / dataset_cfg.DATA_PATH ).resolve()

        print(ROOT_DIR)
        #***'Bus': 0, 'Car': 1, 'Truck': 2, 'Van': 3,'Pedestrian':4,'Cyclist':5,'Trafficcone':6,'Motorcyclist':7 'Tricyclist':8,'Barrowlist':9
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Bus', 'Car', 'Truck', 'Van','Pedestrian','Cyclist','Trafficcone','Motorcyclist','Tricyclist','Barrowlist'],
            data_path= ROOT_DIR ,
            save_path= ROOT_DIR
        )
