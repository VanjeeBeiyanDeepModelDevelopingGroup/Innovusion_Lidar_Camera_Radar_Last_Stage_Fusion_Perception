import os
import numpy as np
import open3d as o3d
import pcl
# class Point(object):
#      def __init__(self,x,y,z):
#            self.x = x
#            self.y = y
#            self.z = z

data = r'/data/Radar_Data/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-velodyne'
save_dir= r'/data/Radar_Data/Baidu_DAIR_V2X_Dataset/RoadSide_DAIR_V2X_I/single-infrastructure-side-example/bin/'
def pcd2bin(filename,save_path):
    points=[]
    points = pcl.load_XYZI(data+'/'+filename).to_array()#[:,:4]
    # points = o3d.io.read_point_cloud(data+'/'+filename)
    # points = np.fromfile(data+'/'+filename,dtype=np.float32).reshape(-1,3)
    # import pdb;pdb.set_trace()
    # with open(data+'/'+filename,'r') as f:
    #    for line in f.readlines()[11:len(f.readlines())-1]: 
    #       strs = line.split(' ')
    #       print(strs)
    #       if len(strs[0]) < 0:
    #           continue
    #       points.append([float(strs[0]),float(strs[1]),float(strs[2]),float(strs[3].strip())])
    # p = np.array(points,dtype=np.float32)
    print(points.shape)
    
    points.tofile(save_path)


pcd_list = os.listdir(data)

for i in pcd_list:
   filename =   i
   pcd2bin(filename, save_dir+i.replace('.pcd', '.bin'))
print('pcd2bin done')