import numpy as np
import json

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        # lines = f.readlines()
        lines = json.load(f)
    objects = [Object3d(line) for line in lines]
    return objects

#'bus', 'car','bicycle','pedestrian', 'tricycle', 'semitrailer','truck'
def cls_type_to_id(cls_type):
    type_to_id = {'Bus': 0, 'Car': 1, 'Truck': 2, 'Van': 3,'Pedestrian':4,'Cyclist':5,'Trafficcone':6,'Motorcyclist':7,'Tricyclist':8,'Barrowlist':9}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

def id_to_cls_type(cls_id):
    type_to_id = {'0': 'bus', '1': 'car', '2':'bicycle', '3':'pedestrian','4':'tricycle','5':'semitrailer','6':'truck'}
    if cls_id not in type_to_id.keys():
        return 'semitrailer'
    return type_to_id[cls_id]

# label有的信息: frameid, clsid, cls_type, angle, h, w, l, x, y, z
class Object3d(object):
    def __init__(self, line):
        label = line
        self.src = line
        
        self.cls_type = label['type']
        self.cls_id = cls_type_to_id(self.cls_type)
        # self.id = label[0]
        import math
        self.alpha = float(label['rotation'])
    
        self.h = float(label['3d_dimensions']['h'])
        self.w = float(label['3d_dimensions']['w'])
        self.l = float(label['3d_dimensions']['l'])
        self.loc = np.array((float(label['3d_location']['x']), float(label['3d_location']['y']), float(label['3d_location']['z'])), dtype=np.float32)
        self.x = float(label['3d_location']['x'])
        self.y = float(label['3d_location']['y'])
        self.z = float(label['3d_location']['z'])

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x , y, z = self.x, self.y, self.z
        theta = self.alpha
        rotMat2D = np.array([[np.sin(theta),np.cos(theta)],[-np.cos(theta),np.sin(theta)]])
        inputArr = np.array([x,y]).reshape(2,-1)
        leftUp_corners = np.array([-w/2,-h/2]).reshape(2,-1)
        rightUP_corners = np.array([w/2,-h/2]).reshape(2,-1)
        leftDown_corners = np.array([-w/2,h/2]).reshape(2,-1)
        rightDown_corners = np.array([w/2,h/2]).reshape(2,-1)
        corner_lu = np.matmul(rotMat2D,leftUp_corners)+inputArr
        corner_ru = np.matmul(rotMat2D,rightUP_corners)+inputArr
        corner_ld = np.matmul(rotMat2D,leftDown_corners)+inputArr
        corner_rd = np.matmul(rotMat2D,rightDown_corners)+inputArr
        corner_lu = corner_lu.reshape(-1)
        corner_ru = corner_ru.reshape(-1)
        corner_ld = corner_ld.reshape(-1)
        corner_rd = corner_rd.reshape(-1)
        corners3d = np.array([corner_lu[0], corner_ru[0], corner_ld[0], corner_rd[0],corner_lu[1], corner_ru[1], corner_ld[1], corner_rd[1]]).reshape(2,-1)
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f   hwl: [%.3f %.3f %.3f] pos: %s ' \
                     % (self.cls_type, self.id, self.alpha, self.h, self.w, self.l,
                        self.loc)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f  %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' \
                    % (self.cls_type, self.id,  self.alpha, 
                        self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2]
                )
        return kitti_str
