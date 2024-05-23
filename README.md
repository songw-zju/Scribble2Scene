# Scribble2Scene
> Song Wang, Jiawei Yu, Wentong Li, Hao Shi, Kailun Yang, Junbo Chen*, Jianke Zhu*

This is the official implementation of **Label-efficient Semantic Scene Completion with Scribble Annotations** (IJCAI 2024)  [[Paper]()] [[Video]()].



## Abstract
Semantic scene completion aims to infer the 3D geometric structures with semantic classes from camera or LiDAR, which provide essential occupancy information in autonomous driving. Prior endeavors concentrate on constructing the network or benchmark in a fully supervised manner. While the dense occupancy grids need point-wise semantic annotations, which incur expensive and tedious labeling costs. In this paper, we build a new label-efficient benchmark, named ScribbleSC, where the sparse scribble-based semantic labels are combined with dense geometric labels for semantic scene completion. In particular, we propose a simple yet effective approach called Scribble2Scene, which bridges the gap between the sparse scribble annotations and fully-supervision. Our method consists of geometric-aware auto-labelers construction and online model training with an offline-to-online distillation module to enhance the performance. Experiments on SemanticKITTI demonstrate that Scribble2Scene achieves competitive performance against the fully-supervised counterparts, showing 99% performance of the fully-supervised models with only 13.5% voxels labeled.


## Framework
<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>



## Citations
```
@inproceedings{wang2024label,
      title={Label-efficient Semantic Scene Completion with Scribble Annotations},
      author={Wang, Song and Yu, Jiawei and Li, Wentong and Shi, Hao and Yang, Kailun and Chen, Junbo and Zhu, Jianke},
      booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI)},
      year={2024}
}
```
