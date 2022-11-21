#!/usr/bin/env python
# encoding=utf8 

import sys
import logging
import pcl
import re

if __name__ == "__main__":

    if sys.argv.__len__() != 3:
        print("Usage: pyton ./BA.py source.pcd dest.pcd")
        print("Auto adjust the type of file: binary or ascii")
        exit(0)
        
    pcl_src_path = sys.argv[1]
    pcl_dest_path = sys.argv[2]

    p = re.compile("DATA (?P<data_type>\w*)")
    data_type = "unknow"
    with open(pcl_src_path, 'rt') as pcl_src_handle:    
        lines = pcl_src_handle.readlines()            

        for index in range(12):
            m = p.match(lines[index])
            try:
                data_type = m.group("data_type")
                break
            except Exception as e:
                pass

    print(data_type)

    if data_type == "binary":
        print("Convert binary to ascii")
        B = False
    elif data_type == "ascii":
        print("Convert ascii to binary")
        B = True
    else:
        print("Usage: pyton ./BA.py source.pcd dest.pcd")
        print("Usage: pyton ./BA.py source.pcd ascii.pcd")
        print("Usage: pyton ./BA.py ascii.pcd binary.pcd")

    pcl_file = pcl.PointCloud_PointXYZI()
    pcl_file.from_file(pcl_src_path)
    pcl_file._to_pcd_file(pcl_dest_path, B)
