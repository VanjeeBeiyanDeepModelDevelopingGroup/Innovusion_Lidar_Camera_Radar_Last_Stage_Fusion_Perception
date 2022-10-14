# training
CUDA_VISIBLE_DEVICES=1\
nohup python3 -u -X faulthandler \
      tools/train.py --cfg_file /data/lianghao/lidar_and_4D_imaging_radar_fusion_demo/pcdet_cfg/models/pointpillar_mf_precat.yaml --batch_size  2