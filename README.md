# SLAM Project: Camera Pose Estimation and 3D Scene Reconstruction

This repository contains the implementation of a Simultaneous Localization and Mapping (SLAM) project aimed at reconstructing the 3D structure of a scene from images taken with a moving camera. This educational project demonstrates the principles of SLAM using basic methods.

## Project Description

The task involves reconstructing the camera positions and the 3D structure of the scene using a set of images captured by a single moving camera. This technique is commonly known as "Structure-from-Motion" (SfM) or "Simultaneous Localization and Mapping" (SLAM). The project utilizes monocular SLAM, focusing on images from a single camera.

### Key Steps

1. **Feature Detection and Description**: Detect keypoints and compute ORB descriptors for each image. Save these descriptors to avoid recomputation.
2. **Feature Matching**: Match keypoints between pairs of reference images and filter incorrect matches using the Fundamental matrix and RANSAC algorithm.
3. **Track Building**: Construct tracks for keypoints across multiple images to represent the same 3D point in different views.
4. **3D Point Triangulation**: Triangulate the 3D coordinates of points using the known positions of reference cameras.
5. **Reprojection Error Calculation**: Project 3D points onto images and compute the reprojection error to filter out inaccurate points.
6. **Pose Estimation**: Estimate the poses of unknown cameras using the matched keypoints and 3D points.
7. **Save Results**: Save the estimated camera positions in the required format.

## Usage

Prepare the Dataset: Place your images and calibration data in the appropriate folders. The structure should follow the format specified in the task description.
Run the Algorithm: Execute the estimate_trajectory.py script with the path to the input data and the output directory.

python estimate_trajectory.py /path/to/input/data /path/to/output/folder

Evaluate the Results: Use the provided evaluation scripts to compare the estimated camera positions with the ground truth.

## Project Structure

estimate_trajectory.py: Main script to implement the SLAM algorithm.
run.py: Script to test the solution with the provided datasets.
generate_point_cloud.py: Script to generate a point cloud from the estimated trajectory.
common/: Directory containing utility scripts for dataset handling, intrinsic parameters, and trajectory management.

## Data Format

rgb.txt: List of images with their relative paths.
known_poses.txt: Known camera poses for some images.
intrinsics.txt: Camera intrinsic parameters.
Evaluation Criteria

This project is intended for educational purposes and personal development in the field of computer vision and SLAM. Feel free to explore, modify, and extend the code to better understand the underlying concepts.

## References

OpenCV Documentation: OpenCV
SLAM Overview Article: Past, Present, and Future of Simultaneous Localization And Mapping
TU Munich RGB-D SLAM Dataset: Dataset

## License

This project is licensed under the Apache-2.0 license. See the LICENSE file for details.
