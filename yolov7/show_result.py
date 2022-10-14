# -- coding: utf-8 --
import cv2
import os


if __name__=="__main__":
    result_path = './runs/detect/exp3'
    fname_list = os.listdir(result_path)
    new_ordered_fname_list = []
    for fname in fname_list:
        print(fname.split(".imagebin.jpg")[0])
        name_order = fname.split(".png")[0]
        name_order_minutes = name_order.split(".")[0]
        name_order_seconds = name_order.split(".")[1]
        new_ordered_fname_list.append(name_order_minutes*1000000+name_order_seconds)
    for new_fname in new_ordered_fname_list:
        img_full_fname = os.path.join(result_path, new_fname)
        img=cv2.imread(img_full_fname)
        cv2.imshow('result', img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()