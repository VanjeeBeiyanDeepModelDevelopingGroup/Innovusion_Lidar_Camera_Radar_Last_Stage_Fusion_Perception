import io as sysio
import torch
import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval
from ....ops.iou3d_nms import iou3d_nms_utils

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """
    根据current_class和difficulty对gt和dt的box进行分类标注并计算有效box数量和dc_bboxes
    Args:
        gt_anno:单帧点云标注dict
        dt_anno:单帧点云预测dict
        current_class: 标量 0
        difficulty: 标量 0
    Returns:
        num_valid_gt:有效gt的数量
        ignored_gt: gt标志列表 0有效,1忽略,-1其他
        ignored_dt:dt标志列表
        dc_bboxes: don't care的box
    """
    CLASS_NAMES = ['car','bicycle', 'bus', 'tricycle', 'pedestrian', 'semitrailer', 'truck']
    # CLASS_NAMES = ['car','bicycle', 'tricycle', 'pedestrian', 'bus', 'semitrailer', 'truck']
    MIN_HEIGHT = [40, 25, 25] # 最小高度阈值
    MAX_OCCLUSION = [0, 1, 2] # 最大遮挡阈值
    MAX_TRUNCATION = [0.15, 0.3, 0.5] # 最大截断阈值
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    ###################### modified
    num_valid_dt = 0
    gt_boxes = []
    dt_boxes = []
    # 2.遍历所有gt框
    for i in range(num_gt):
        # 获取第i个bbox，name和height等信息
        box_gt = np.concatenate(
            [gt_anno["location"][i], gt_anno["dimensions"][i], gt_anno["alpha"][..., np.newaxis][i]], 0)

        gt_name = gt_anno["name"][i].lower()

        valid_class = -1
        # 2.1首先，根据类别进行三个判断给类别分类: 1有效，0忽略，-1无效
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            gt_boxes.append(box_gt)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        # for i in range(num_gt):
        
        if gt_anno["name"][i] == "unknown":
            dc_bboxes.append(np.concatenate([gt_anno["location"][i][:2], gt_anno["dimensions"][i][:2]]))
    for i in range(num_dt):
        box_dt = np.concatenate(
            [dt_anno["location"][i], dt_anno["dimensions"][i], dt_anno["alpha"][..., np.newaxis][i]], 0)
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        if valid_class == 1:
            ignored_dt.append(0)
            dt_boxes.append(box_dt)
            num_valid_dt += 1
        else:
            ignored_dt.append(-1)

    return num_valid_gt, num_valid_dt, ignored_gt, ignored_dt, dc_bboxes, gt_boxes, dt_boxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    """
    逐帧计算tp, fp, fn, similarity, thresholds[:thresh_idx]等统计指标
    根据compute_fp的状态不同,存在两种模式
    Args:
        overlaps:单帧点云的iou(N,M)\n
        gt_datas:(N,5)--> (x1, y1, x2, y2, alpha)\n
        dt_datas:(M,6)--> (x1, y1, x2, y2, alpha, score)\n
        ignored_gt:(N,)为box的状态 0,1,-1\n
        ignored_det:(M,)为box的状态 0,1,-1\n
        dc_bboxes:(k,4)\n
        metric:0: bbox, 1: bev, 2: 3d\n
        min_overlap:最小iou阈值\n
        thresh=0:忽略score低于此值的dt,根据recall点会传入41个阈值\n
        compute_fp=False\n
        compute_aos=False\n
    Returns:
        tp: 真正例 预测为真，实际为真\n
        fp: 假正例 预测为真，实际为假\n
        fn: 假负例 预测为假，实际为真\n
        similarity:余弦相似度\n
        thresholds[:thresh_idx]:与有效gt匹配dt分数\n
    precision = TP / TP + FP 所有预测为真中,TP的比重\n
    recall = TP / TP + FN 所有真实为真中,TP的比重\n
    """
    # ============================ 1 初始化============================
    det_size = dt_datas.shape[0] # det box的数量 M
    gt_size = gt_datas.shape[0] # gt box的数量 N 
    dt_scores = dt_datas[:, -1] # dt box的得分 (M,)
    dt_alphas = dt_datas[:, 4]     # dt alpha的得分 (M,) # rotation_y
    gt_alphas = gt_datas[:, 4] # gt alpha的得分 (N,)
    dt_bboxes = dt_datas[:, :4] # (M,4)
    gt_bboxes = gt_datas[:, :4]  # (N,4)

    # 该处的初始化针对dt
    assigned_detection = [False] * det_size # 存储dt是否匹配了gt
    ignored_threshold = [False] * det_size # 如果dt分数低于阈值，则标记为True
    # 如果计算fp: 预测为真，实际为假
    if compute_fp:
        # 遍历dt
        for i in range(det_size):
            # 如果分数低于阈值
            if (dt_scores[i] < thresh):
                # 忽略该box
                ignored_threshold[i] = True

    # 初始化
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0 # thresholds的index，后面更新
    delta = np.zeros((gt_size, ))
    delta_idx = 0 # delta的index，后面更新

    # ============================ 2 针对gt找匹配的dt，同时计算tp和fn，因为这是针对gt的============================
    # 遍历gt，这里还是以gt为主，针对gt找匹配的dt，跳过不合格的gt的dt
    for i in range(gt_size):
        if ignored_gt[i] == -1: # 跳过无效gt
            continue
        det_idx = -1 # 储存目前为止匹配dt的最佳idx
        valid_detection = NO_DETECTION # 标记是否为有效dt
        max_overlap = 0 # 存储到目前为止匹配dt的最佳overlap
        assigned_ignored_det = False # 标记是否匹配上dt

        # 遍历dt
        for j in range(det_size):
            if (ignored_det[j] == -1): # 跳过无效dt
                continue
            if (assigned_detection[j]): # 如果已经匹配了gt，跳过
                continue
            if (ignored_threshold[j]):  # 如果dt分数低于阈值，则跳过
                continue
            overlap = overlaps[j, i]  # 获取当前dt和此gt之间的overlap 
            dt_score = dt_scores[j]  # 获取当前dt的分数

            if (not compute_fp and (overlap > min_overlap) # compute_fp为false，不需要计算FP
                    and dt_score > valid_detection): # overlap大于最小阈值比如0.7
                det_idx = j
                valid_detection = dt_score # 更新到目前为止检测到的最大分数
            elif (compute_fp and (overlap > min_overlap)  # 当compute_fp为true时，基于overlap进行选择 overlap要大于最小值
                  and (overlap > max_overlap or assigned_ignored_det) # 如果当前overlap比之前的最大overlap还大或者gt以及匹配dt
                  and ignored_det[j] == 0):  # dt有效
                max_overlap = overlap  # 更新最佳的overlap
                det_idx = j # 更新最佳匹配dt的id
                valid_detection = 1 # 标记有效dt
                assigned_ignored_det = False # 用留一法来表明已经分配了关心的单位
            elif (compute_fp and (overlap > min_overlap) # compute_fp为true 如果重叠足够
                  and (valid_detection == NO_DETECTION) # 尚未分配任何东西
                  and ignored_det[j] == 1): # dt被忽略
                det_idx = j # 更新最佳匹配dt的id
                valid_detection = 1 # 标记有效dt
                assigned_ignored_det = True # 标志gt已经匹配上dt
        # 如果有效gt没有找到匹配，则fn加一，因为真实标签没有找到匹配
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        # 如果gt找到了匹配，并且gt标志为忽略或者dt标志为忽略，则assigned_detection标记为True
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        # 否则有效gt找到了匹配
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True

    # ============================ 3 计算fp，这是针对dt的 ============================
    # 在遍历完全部的gt和dt后，如果compute_fp为真，则计算fp
    if compute_fp:
        # 遍历dt
        for i in range(det_size):
            # 如果以下四个条件全部为false，则fp加1
            # assigned_detection[i] == 0 --> dt没有分配gt
            # ignored_det[i] != -1 and ignored_det[i] ！= 1 --> gnored_det[i] == 0 有效dt
            # ignored_threshold[i] == false, 无法忽略该dt box
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1 # 预测为真，实际为假
        nstuff = 0
        if metric == 0: # 如果计算的是bbox
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            # fp+tp(有效gt找到了匹配）= 在该recall阈值下的所有预测为真的box数量
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    """
    计算part的pr
    Args:
        overlaps: 一个part的iou (M,N)-->(642,233)
        pr: (41,4)--> tp, fp, fn, similarity
        gt_nums: 一个part的gt的数量 (37,)
        dt_nums: 一个part的dt的数量 (37,)
        dc_nums: 一个part的dc的数量 (37,)
        gt_datas: 一个part的gt的数据 (233,5)
        dt_datas: 一个part的gt的数据 (642,5)
        dontcares: 一个part的gt的数据 (79,4)
        ignored_gts: (233,)
        ignored_dets: (642,)
        metric: 0
        min_overlap: 0.7
        thresholds: (41,)
        compute_aos=False: True
    Return:
        传入的参数有pr,因此没有该函数没有返回值,返回值在参数中
    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    # 遍历part点云，逐帧计算累加pr和box数量
    for i in range(gt_nums.shape[0]):
        # 遍历阈值
        for t, thresh in enumerate(thresholds):
            # 提取该帧点云的iou矩阵
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]
            # 取出该帧的数据
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            # 真正计算指标
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            # 累加计算指标
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        # 累加box数量
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]
        
import pcdet.box_np_ops as box_np_ops
def rotate_nms_cc_new(dets,trackers):
    trackers_corners = box_np_ops.center_to_corner_box2d(trackers[:, :2], trackers[:, 3:5], trackers[:, 6])
    trackers_standup = box_np_ops.corner_to_standup_nd(trackers_corners)
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 3:5],dets[:, 6])
    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)
    standup_iou = box_np_ops.iou_jit(dets_standup, trackers_standup, eps=0.0)
    return standup_iou

def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    # 1.计算每一帧点云的box数量
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    # 2.按照part计算指标
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part] # 取6帧点云
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        # 根据metric的index分别计算不同指标
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        # 在相机坐标系下进行计算，z轴朝前，y轴朝下，x轴朝右，因此，俯视图取x和z
        # bev
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["alpha"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["alpha"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        # 3d
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["alpha"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["alpha"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            # overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
            #     np.float64)
            overlap_part = rotate_nms_cc_new(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    # 前面进行了多余的计算（一个part内的不同点云之间也计算了overlap），这里进行截取   
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        # 逐帧计算overlaps
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            # 在overlaps的第j个part中根据gt_box_num和dt_box_num截取对应的iou
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    # overlaps共631个元素，每个元素代表了一帧点云的iou 例如：overlaps[0].shape=(41, 39)
    # parted_overlaps共101个元素，表示每个part的iou  例如：parted_overlaps[0].shape = (246, 238)
    # total_gt_num共631个元素，表示每帧点云的gt_box的数量  total_gt_num[0] = 41
    # total_dt_num共631个元素，表示每帧点云的det_box的数量  total_dt_num[0] = 39
    # import pdb;pdb.set_trace()
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    total_num_valid_dt = 0  ##################################modify
    recall_dict = {}
    recall_dict = {'gt': 0}
    recall_dict['rcnn_%s' % (str(current_class))] = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, num_valid_dt, ignored_gt, ignored_det, dc_bboxes, gt_boxes, dt_boxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        total_num_valid_dt += num_valid_dt
        # gt_data = np.concatenate(
        #     [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        # dt_data = np.concatenate([
        #     dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
        #     dt_annos[i]["score"][..., np.newaxis]
        # ], 1)
        gt_data = np.concatenate(
            [gt_annos[i]["location"][:,:2], gt_annos[i]["dimensions"][:,:2],
             gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_data = np.concatenate(
            [dt_annos[i]["location"][:,:2], dt_annos[i]["dimensions"][:,:2],
             dt_annos[i]["alpha"][..., np.newaxis], dt_annos[i]["score"][..., np.newaxis]], 1)

        gt_boxes = np.array(gt_boxes)
        dt_boxes = np.array(dt_boxes)

        gt_datas = torch.from_numpy(gt_boxes)
        gt_datas = gt_datas.to(torch.float32)
        dt_datas = torch.from_numpy(dt_boxes)
        dt_datas = dt_datas.to(torch.float32)
        if gt_datas.shape[0] > 0:
            if dt_datas.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(gt_datas.cuda(), dt_datas[:, 0:7].cuda())
            else:
                iou3d_rcnn = torch.zeros((0, gt_datas.shape[0]))

            if iou3d_rcnn.shape[0] == 0:
                recall_dict['rcnn_%s' % str(current_class)] += 0
            else:
                ################################ iou thresth
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > 0.25).sum().item()
                recall_dict['rcnn_%s' % str(current_class)] += rcnn_recalled
            recall_dict['gt'] += gt_datas.shape[0]

        gt_datas_list.append(gt_data)
        dt_datas_list.append(dt_data)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt, total_num_valid_dt, recall_dict)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos) # 631
    # 1、split_parts共101个元素，100个6和一个31--> [6,6......31]
    split_parts = get_split_parts(num_examples, num_parts) 
    # import pdb;pdb.set_trace()
    # 2.计算iou
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    # overlaps共631个元素，每个元素代表了一帧点云的iou 例如：overlaps[0].shape=(41, 39)
    # parted_overlaps共101个元素，表示每个part的iou  例如：parted_overlaps[0].shape = (246, 238)
    # total_gt_num共631个元素，表示每帧点云的gt_box的数量  total_gt_num[0] = 41
    # total_dt_num共631个元素，表示每帧点云的det_box的数量  total_dt_num[0] = 39
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    # 3.计算初始化需要的数组维度
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps) #2
    num_class = len(current_classes) # 7
    num_difficulty = len(difficultys) #  1
    # 初始化precision,recall和aos
    # precision = (7, 1, 2, 41)
    precision = np.zeros(   
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    ##############################################33
    precision_all = np.zeros(
        [num_minoverlap, N_SAMPLE_PTS])
    # recall = np.zeros(
    #     [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty])  ##################################modify
    recall_all = np.zeros([num_class])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    aos_all = np.zeros([num_minoverlap, N_SAMPLE_PTS])

    all_thresholds = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    gt_num = np.zeros([num_class])
    dt_num = np.zeros([num_class])

    # pr_all = np.zeros([25, 4])
    # 4、逐类别遍历 7
    for m, current_class in enumerate(current_classes):
        # 逐难度遍历 0: easy, 1: normal, 2: hard
        # 这里只有一个难度
        for l, difficulty in enumerate(difficultys):
            # 4.1 数据准备
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            # gt_datas_list: 631个元素，每个元素为(N,5)--> (x1, y1, x2, y2, alpha) 注意N不相等
            # dt_datas_list: 631个元素，每个元素为(M,6)--> (x1, y1, x2, y2, alpha, score) 注意M不相等
            # ignored_gts: 631个元素,每个元素为（N,）为每个box的状态 0，1，-1
            # ignored_dets: 631个元素,每个元素为（M,）为每个box的状态 0，1，-1
            # dontcares: 631个元素，每个元素为(k,4) 注意K不相等
            # total_dc_num: 631个元素，表示每帧点云dc box的数量
            # total_num_valid_gt:全部有效box的数量: 1589
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt, total_num_valid_dt, recall_dict) = rets
             # 4.2 逐min_overlap遍历
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                # 4.2.1 计算所有有效gt匹配上的dt的全部分数，共2838个box score和41个recall阈值点
                for i in range(len(gt_annos)):
                    # 这里调用compute_statistics_jit没有真正计算指标，而是获取thresholdss，然后计算41个recall阈值点
                    rets = compute_statistics_jit(
                        overlaps[i], # 单帧点云的iou（N,M）
                        gt_datas_list[i],  # (N,5)--> (x1, y1, x2, y2, alpha)
                        dt_datas_list[i], # (M,6)--> (x1, y1, x2, y2, alpha, score)
                        ignored_gts[i], # （N,）为box的状态 0，1，-1
                        ignored_dets[i], # （M,）为box的状态 0，1，-1
                        dontcares[i], # (k,4)
                        metric,  # 0: bbox, 1: bev, 2: 3d
                        min_overlap=min_overlap, # 最小iou阈值
                        thresh=0.0, # 忽略score低于此值的dt
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)

                all_thresholds[m, l, k, :len(thresholds)] = thresholds  #############################modify
                gt_num[m] = total_num_valid_gt
                dt_num[m] = total_num_valid_dt

                pr = np.zeros([len(thresholds), 4])
                idx = 0
                # 4.2.2 遍历101个part，在part内逐帧逐recall_threshold计算tp, fp, fn, similarity，累计pr
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    # 真正计算指标，融合统计结果
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                # 4.2.3 根据不同类别，难度和最小iou阈值以及recall阈值，计算指标
                # pr_all += pr   ##############################
                for i in range(len(thresholds)):
                    # recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    # m:类别，l:难度，k:min_overlap, i:threshold
                    # pr:（41，4）--> tp, fp, fn, similarity
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                # 4.2.4 因为pr曲线是外弧形，按照threshold取该节点后面的最大值，相当于按照节点截取矩形
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    # recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
            if recall_dict['gt'] == 0:
                recall[m, l] = 0
            else:
                recall[m, l] = recall_dict['rcnn_%s' % str(current_class)] / recall_dict['gt']

    # for k, min_overlap in enumerate(min_overlaps[:, metric, 0]):
    #     for i in range(25):
    #         precision_all[k, i] = pr_all[i, 0] / (pr_all[i, 0] + pr_all[i, 1])
    #         if compute_aos:
    #             aos_all[k, i] = pr_all[i, 3] / (pr_all[i, 0] + pr_all[i, 1])
    #     for i in range(25):
    #         precision_all[k, i] = np.max(
    #             precision_all[k, i:], axis=-1)
    #         if compute_aos:
    #             aos_all[k, i] = np.max(aos_all[0, k, i:], axis=-1)

    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
        "thresholds": thresholds,
        "min_overlaps": min_overlaps,
        "gt_num": gt_num,
        "dt_num": dt_num,
        # "precision_all": precision_all,
        # "orientation_all": aos_all
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0]
    metric = {}  ##################################modify
    ret_bev = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps, compute_aos)

    ret_3d = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps, compute_aos)
    # mAP_3d = get_mAP(ret["precision"])
    # mAP_3d_R40 = get_mAP_R40(ret["precision"])
    # if PR_detail_dict is not None:
    #     PR_detail_dict['3d'] = ret['precision']
    # return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40

    metric["bev"] = ret_bev
    metric["3d"] = ret_3d
    return metric

def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_mod = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    overlap_easy = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                             [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    min_overlaps = np.stack([overlap_mod, overlap_easy], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'car',
        1: 'bicycle',
        2: 'bus',
        3: 'tricycle',
        4: 'pedestrian',
        5: 'semitrailer',
        6: 'truck'
    }
    # 'car','bicycle', 'tricycle', 'pedestrian', 'bus', 'semitrailer', 'truck'
    # class_to_name = {
    #     0: 'car',
    #     1: 'bicycle',
    #     2: 'tricycle',
    #     3: 'pedestrian',
    #     4: 'bus',
    #     5: 'semitrailer',
    #     6: 'truck'
    # }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            # if anno['rotation_y'][0] != -10:
            compute_aos = True
            # break
    # mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
    #     gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)
    metric = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    # ret_dict = {}
    detail = {}  ###################################modify
    # for j, curcls in enumerate(current_classes):
        # # mAP threshold array: [num_minoverlap, metric, class]
        # # mAP result: [num_class, num_diff, num_minoverlap]
        # for i in range(min_overlaps.shape[0]):
        #     result += print_str(
        #         (f"{class_to_name[curcls]} "
        #          "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
        #     result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
        #                          f"{mAPbbox[j, 1, i]:.4f}, "
        #                          f"{mAPbbox[j, 2, i]:.4f}"))
        #     result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
        #                          f"{mAPbev[j, 1, i]:.4f}, "
        #                          f"{mAPbev[j, 2, i]:.4f}"))
        #     result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
        #                          f"{mAP3d[j, 1, i]:.4f}, "
        #                          f"{mAP3d[j, 2, i]:.4f}"))
        #
        #     if compute_aos:
        #         result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
        #                              f"{mAPaos[j, 1, i]:.2f}, "
        #                              f"{mAPaos[j, 2, i]:.2f}"))
        #         # if i == 0:
        #            # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
        #            # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
        #            # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]
        #
        #     result += print_str(
        #         (f"{class_to_name[curcls]} "
        #          "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
        #     result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
        #                          f"{mAPbbox_R40[j, 1, i]:.4f}, "
        #                          f"{mAPbbox_R40[j, 2, i]:.4f}"))
        #     result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
        #                          f"{mAPbev_R40[j, 1, i]:.4f}, "
        #                          f"{mAPbev_R40[j, 2, i]:.4f}"))
        #     result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
        #                          f"{mAP3d_R40[j, 1, i]:.4f}, "
        #                          f"{mAP3d_R40[j, 2, i]:.4f}"))
        #     if compute_aos:
        #         result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
        #                              f"{mAPaos_R40[j, 1, i]:.2f}, "
        #                              f"{mAPaos_R40[j, 2, i]:.2f}"))
        #         if i == 0:
        #            ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
        #            ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
        #            ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]
        #
        #     if i == 0:
        #         # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
        #         # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
        #         # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
        #         # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
        #         # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
        #         # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
        #         # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
        #         # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
        #         # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]
        #
        #         ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
        #         ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
        #         ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
        #         ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
        #         ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
        #         ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
        #         ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
        #         ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
        #         ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    # return result, ret_dict
    for j, curcls in enumerate(current_classes):
        class_name = class_to_name[curcls]
        detail[class_name] = {}
        detail[class_name][f"gt_num"] = int(metric["3d"]["gt_num"][j])
        detail[class_name][f"dt_num"] = int(metric["3d"]["dt_num"][j])
        detail[class_name][f"recall"] = (np.around(metric["3d"]["recall"][j] * 100, 2)).tolist()

        for i in range(min_overlaps.shape[0]):
            mAP3d = get_mAP(metric["3d"]["precision"][j, :, i])
            detail[class_name][f"3d@{min_overlaps[i, 2, j]:.2f}"] = np.around(mAP3d, 2).tolist()
            result += print_str(
                (f"{class_to_name[curcls]} ", "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(f"3d   AP:{mAP3d}")

            ################################# bev ###########################################
            mAPbev = get_mAP(metric["bev"]["precision"][j, :, i])
            detail[class_name][f"bev@{min_overlaps[i, 2, j]:.2f}"] = np.around(mAPbev, 2).tolist()
            result += print_str(
                (f"{class_to_name[curcls]} ",
                 "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            result += print_str(f"bev   AP:{mAPbev}")

            # ################################### aos ###########################################
            # mAPaos = get_mAP(metric["3d"]["orientation"][j, :, i])
            # detail[class_name][f"aos@{min_overlaps[i, 2, j]:.2f}"] = np.around(mAPaos, 2).tolist()
            # result += print_str(
            #     (f"{class_to_name[curcls]} ",
            #      "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            # mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
            # result += print_str(f"aos   AP:{mAPaos}")
            #
            # ################################# bev aos ###########################################
            # mAPaos_bev = get_mAP(metric["bev"]["orientation"][j, :, i])
            # detail[class_name][f"bev aos@{min_overlaps[i, 2, j]:.2f}"] = np.around(mAPaos_bev, 2).tolist()
            # result += print_str(
            #     (f"{class_to_name[curcls]} ",
            #      "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            # mAPaos_bev = ", ".join(f"{v:.2f}" for v in mAPaos_bev)
            # result += print_str(f"bev aos   AP:{mAPaos_bev}")

    # for i in range(min_overlaps.shape[0]):
    #     mAP3d_all = get_mAP(metric["3d"]["precision_all"][:, i])
    #     mAP3d_all = ", ".join(f"{v:.2f}" for v in mAP3d_all)
    #     result += print_str(f"mAP 3d  AP:{mAP3d_all}")
    #
    # for i in range(min_overlaps.shape[0]):
    #     bev_all = get_mAP(metric["bev"]["precision_all"][:, i])
    #     bev_all = ", ".join(f"{v:.2f}" for v in bev_all)
    #     result += print_str(f"mAP bev  AP:{bev_all}")
    #
    # for i in range(min_overlaps.shape[0]):
    #     aos_3d_all = get_mAP(metric["3d"]["orientation_all"][:, i])
    #     aos_3d_all = ", ".join(f"{v:.2f}" for v in aos_3d_all)
    #     result += print_str(f"mAP 3d aos  AP:{aos_3d_all}")
    #
    # for i in range(min_overlaps.shape[0]):
    #     aos_bev_all = get_mAP(metric["bev"]["orientation_all"][:, i])
    #     aos_bev_all = ", ".join(f"{v:.2f}" for v in aos_bev_all)
    #     result += print_str(f"mAP bev aos  AP:{aos_bev_all}")

    return {
        "result": result,
        "detail": detail,
    }