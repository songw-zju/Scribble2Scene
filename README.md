# Scribble2Scene
> Song Wang, Jiawei Yu, Wentong Li, Hao Shi, Kailun Yang, Junbo Chen*, Jianke Zhu*

This is the official implementation of **Label-efficient Semantic Scene Completion with Scribble Annotations** (IJCAI 2024)  [[Paper](https://arxiv.org/pdf/2405.15170.pdf)].

<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>


## Getting Started

We provide the core codes of our proposed Scribble2Scene for online model training (Stage-II):

```
./code
    └── projects/
    │       ├── configs/
    │       │     ├── scribble2scene/
    |       |     |          ├──scribble2scene-distill.py  # the config file for Scribble2Scene Stage-II
    │       ├── mmdet3d_plugin/
    │       │     ├── scribble2scene/
    |       |     |          ├──detectors
    |       |     |          |    ├──scribble2scene_distill.py  # our Teacher-Labeler and online model architecture
    |       |     |          ├──dense_heads
    |       |     |          |    ├──scribble2scene_head.py  # our used completion head and loss functions
    |       |     |          ├──utils
    |       |     |          |    ├──distillation_loss.py  # our proposed range-guided offline-to-online distillation loss
    └──tools/
```

### Prepare Data-SemanticKITTI

Direct downloading: 

- The **semantic scene completion dataset v1.1** (SemanticKITTI voxel data, 700 MB) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download).
- The **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Run and Eval

Train the online model **with our proposed Scribble2Scene** on 4 GPUs 

```
./tools/dist_train.sh ./projects/configs/scribble2scene/scribble2scene-distill.py 4
```

Eval the online model **with our proposed Scribble2Scene** on 4 GPUs

```
./tools/dist_test.sh ./projects/configs/scribble2scene/scribble2scene-distill.py ./path/to/ckpts.pth 4
```



## Acknowledgement

Many thanks to these excellent open source projects:

- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
