# Predicting Alignability of Point Cloud Pairs for Point Cloud Registration Using Features

Point cloud registration is often used in fields like SLAM where the overlap of two consecutive point clouds is large. But in fields like multi-sensor fusion of point clouds and LiDAR-based localization, there is a high chance of registering non-overlapping point cloud pairs. Since in such cases, the result will always be a wrong transformation, it is useful to evaluate the alignability of the point cloud pairs prior to the registration. We propose an algorithm that predicts the alignability of two point clouds based on the minimum distances of descriptors. It calculates statistical values describing the minimum distances and classifies the point cloud pairs.

## Installation

### Prerequisites

- C++
 - Point Cloud Library (Version 1.8)
- Python 3
 - scikit-learn==0.24.1
 - pandas==1.2.4
 - matplotlib==3.3.4
 - seaborn==0.11.1

### Build

```bash
mkdir build
cd build
cmake ..
make
```

## Demo

```bash
./build/demo/demo_alignabilityPrediction ./dataset/robot/k4/0/0.ply ./dataset/ceiling/0.ply 1.0
```

## Data set

The data set can be found in the _dataset_ folder. The point clouds are split into target (_ceiling_) and source (_robot_). The source clouds are further split into clouds at different positions with overlap (_k*_) and without overlap (_f*_). For each position, multiple rotations were captured.

The data set has been preprocessed using voxelization with a voxel size of 0.05 m³ and a radius outlier removal with a radius of 0.1 m and a required neighoring point count of four.

## Citation

```
@inproceedings{kirsch2022,
    author = {Kirsch, André and Günter, Andrei and König, Matthias},
    title = {Predicting Alignability of Point Cloud Pairs for Point Cloud Registration Using Features},
    booktitle = {12th International Conference on Pattern Recognition Systems (ICPRS)},
    year = {2022}
}
```
