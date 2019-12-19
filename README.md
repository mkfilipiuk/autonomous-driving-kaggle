##### Preparing Kaggle dataset for 6DVNET
1. ``pip3 install kaggle``
2. Kaggle > my account > generate API key > ``~/.kaggle/kaggle.json``
3. ``cd data``
4. ``bash get_kaggle_dataset_and_also_prepare_as_6dvnet.sh``

##  6 DoF estimation network

### Based on Mask-RCNN

I am afraid that the current repo doesn't generate the winning solution of the ApolloScape 3D car Instance Understanding challenge.
However, a very under-documented repo does have the winning solution code:
https://github.com/stevenwudi/ApolloScape_InstanceSeg

Hope it helps

### Dataset:

- [Pascal3D+](http://cvgl.stanford.edu/projects/pascal3d.html)
- [ApolloScape 3D Car Instance](http://apolloscape.auto/car_instance.html)
- [Kitti](http://www.cvlibs.net/datasets/kitti/) 


### Citation

I would appreciate citation of the following paper if you find this repository helpful.

```
@InProceedings{Wu_2019_CVPR_Workshops,
author = {Wu, Di and Zhuang, Zhaoyong and Xiang, Canqun and Zou, Wenbin and Li, Xia},
title = {6D-VNet: End-To-End 6-DoF Vehicle Pose Estimation From Monocular RGB Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
```
# Run
1. Copy this repository
2. Create virtualenv and install requirements.txt
3. Update maskrcnn_benchmark/config/paths_catalog.py with your dataset's paths
4. Run one of prepared training, for example tools/train_pascal_3d+.py