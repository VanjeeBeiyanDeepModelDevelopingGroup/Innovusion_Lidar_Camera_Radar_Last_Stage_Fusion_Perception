import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pcl
def Color_map():
    num0 = 256
    num1 = 0
    num2 = 0
    ran = 64
    step = 4
    colormap = []

    for i in range(ran):
        num1 += 4
        colormap.append([num0, num1, num2])
    for i in range(ran):
        num0 -= 4
        colormap.append([num0, num1, num2])
    for i in range(ran):
        num2 += 4
        colormap.append([num0, num1, num2])
    for i in range(ran):
        num1 -= 4
        colormap.append([num0, num1, num2])
    return colormap


def pcd_files_find(file_dir):
    pcd_name = []
    # import pdb;pdb.set_trace()
    for root,dirs,files in os.walk(file_dir):
        for file_name in files:
            if file_name.endswith('pcd'):
                pcd_name.append(root+'/'+file_name) 
    return pcd_name

def img_files_find(file_dir):
    img_name = []
    for root,dirs,files in os.walk(file_dir):
        for file_name in files:
            if file_name.endswith('jpg'):
                img_name.append(root+'/'+file_name)
    return img_name

def readpcd(filename):                     # 读取数据，将数组转化为向量
    points=[]
    #读取pcd文件,从pcd的第12行开始是三维点
    with open(filename) as f:
       for line in f.readlines()[11:len(f.readlines())-1]:
          strs = line.split(' ')
          points.append([float(strs[0]),float(strs[1]),float(strs[2]),float(strs[3].strip())])
    pt = np.array(points,dtype=np.float32)
    return pt


def pointpick(pcdata):        # 裁剪点云
    cutPoints = np.delete(pcdata, np.where(pcdata[0, :] > 2048), axis=1)
    cutPoints = np.delete(cutPoints, np.where(cutPoints[0, :] < 0), axis=1)
    cutPoints = np.delete(cutPoints, np.where(cutPoints[1, :] < 0), axis=1)
    cutPoints = np.delete(cutPoints, np.where(cutPoints[1, :] > 1400), axis=1)
    cutPoints[2,:] = np.abs(cutPoints[2,:])-np.min(np.abs(cutPoints[2,:]))
    cutPoints[2, :] = cutPoints[2,:]/np.max(cutPoints[2,:])*255
    return cutPoints


def changeLabel(pcPoint):         # 坐标转换
    fx = 1.2e3
    fy = 1.2e3
    u0 = 1028
    v0 = 753
    lidar2camara = [[0.0021,-0.9798,0.2000,0],[1,-0.0001,-0.0110,0],[0.0108,0.200,0.9797,0],[0.1428,-0.1462,0.2815,1]]
    lidar2virtual =  np.array([[0.978966474533, 0.000000000000, -0.204021081328, 0.000000000000],
                    [0.000000000000, 1, 0, 0.000000000000],
                    [0.204021081328,0,0.978966474533, -5.500000000000],
                    [0.000000000000, 0.000000000000, 0.000000000000 ,1.000000000000]])
    # fx = 2.27e3
    # fy = 2.27e3
    # u0 = 1040
    # v0 = 700
    # lidar2camara = [[0,1,0,0],[-1,0,0,0],[0,0,-1,0],[-0.1,-0.175,-0.01,1]]
    virtual2pixel =  np.array([[0, 0, 1],[0, -1, 0],[1, 0, 0]])
    camara2pixel = [[fx, 0, 0], [0, fy, 0], [u0, v0, 1]]
    Line1, Colum1 = np.shape(pcPoint)
    new_points = np.ones((Line1, Colum1 + 1), dtype=np.float32)
    new_points[:, :3] = pcPoint
    # import pdb;pdb.set_trace()
    # lidar_camara_points = np.dot(new_points,lidar2camara)
    # camara_image_points = np.dot(lidar_camara_points[:, :3], camara2pixel)
    lidar_camara_points = np.matmul(pcPoint,virtual2pixel)
    lidar_camara_points = np.matmul(lidar_camara_points,lidar2virtual[:3,:3]) 
    lidar_camara_points[:,2]+=5.5 
    # camara_image_points = np.dot(lidar_camara_points[:, :3], virtual2pixel)
    # camara_image_points = camara_image_points.swapaxes(0, 1)
    # camara_image_points[:2,:] /= camara_image_points[2,:]
    # cutPoints = pointpick(camara_image_points)
    return lidar_camara_points

def time_syn(lidardata,imagedata):            # 时间同步
    img_cnt = 0
    lidar_cnt = 0
    new_img = []
    new_lidar = []
    for i in range(len(imagedata)):
        img_name = imagedata[img_cnt]
        lidar_name = lidardata[lidar_cnt]
        img_time = img_name[-30:-13]
        lidar_time = lidar_name[-25:-8]
        while float(img_time)>float(lidar_time):
            lidar_name = lidardata[lidar_cnt + 1]
            lidar_time = lidar_name[-25:-8]
            lidar_cnt += 1
        new_img.append(imagedata[img_cnt])
        new_lidar.append(lidardata[lidar_cnt])
        print(float(lidar_time) - float(img_time))
        img_cnt += 1
        lidar_cnt += 1
    return new_lidar,new_img


def drawvidio(lidardata,imagedata):       # 动态显示
    # import pdb;pdb.set_trace()
    # cv2.namedWindow('demo', 0)
    # cv2.resizeWindow('demo', 1024, 700)
    # cmp = Color_map()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1024,height =720)
    import time
    from tqdm import tqdm,trange
    path='/data/chenruiming/virtual_lidar2/'
    pbar = tqdm(lidardata)
    for i in pbar:
        # for i in range(30):

        pbar.set_description('Lidar to virtual lidar')
        filepath, filename = os.path.split(i)
        name,ext = os.path.splitext(filename)
        # import pdb;pdb.set_trace()
        # picdata = cv2.imread(imagedata[i])
        # points = o3d.io.read_point_cloud(i)
        try:
            points = pcl.load_XYZI(i).to_array()
        except:
            pass
        # points = np.array(points.points)
        intensity = points[:,3]
        new_points = changeLabel(points[:,:3])
        
        # print(new_points.shape)
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(new_points)
        # vis.add_geometry(point_cloud)
        # render = vis.get_render_option()
        # render.point_size = 1
        # vis.run()
        # time.sleep(0.1)
        # vis.clear_geometries()
        
        # path_pcd='/data/Radar_Data/test1.pcd'
        points[:,:3]= new_points
        points.tofile(path+str(name))
        
        # o3d.io.write_point_cloud(path+str(filename),point_cloud,write_ascii=True)
        # import pdb;pdb.set_trace()
        # Line1, Colum1 = np.shape(new_points)
        # x = new_points[0, :]
        # y = new_points[1, :]
        # z = new_points[2, :]
        # for j in range(Colum1):
        #     dx = int(x[j])
        #     dy = int(y[j])
        #     dz = int(z[j])
        #     cv2.circle(picdata, (dx, dy), 1, (cmp[255-dz]), thickness=2)
        # cv2.imshow('demo', picdata)
        # cv2.waitKey(40)

if __name__ == '__main__':
    dataset = "/data/chenruiming/camera_lidar/20220927/come/lidar2"
    imageName = img_files_find(dataset)
    lidarName = pcd_files_find(dataset)
    # print(lidarName)
    # import pdb;pdb.set_trace()
    new_lidar, new_img = time_syn(lidarName,imageName)
    drawvidio(lidarName,new_img)

    # print(len(new_img))
    # print(len(new_lidar))
    # print(new_img[-1])
    # print(new_lidar[-1])