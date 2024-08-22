# SLAM (Simultaneous Localization and Mapping) for Scene Structure Reconstruction

This repository contains the implementation of a SLAM-based project, developed as part of the Computer Vision and Image Processing course at the Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University.

## Project Overview

The project focuses on reconstructing the 3D structure of a scene using photographs captured from a moving camera. This task is a classic problem in computer vision, commonly referred to as **Structure-from-Motion (SfM)** or **Simultaneous Localization and Mapping (SLAM)**. The implementation is based on a monocular SLAM approach, which involves a single moving camera.

### Key Topics and Concepts
- **SLAM**: A method that allows for the reconstruction of the 3D structure of a scene and camera positions using images taken from different angles.
- **Feature Extraction**: Using ORB (Oriented FAST and Rotated BRIEF) to detect key points and compute descriptors in the images.
- **Feature Matching**: Matching key points across images using descriptor space and filtering with RANSAC and the fundamental matrix.
- **3D Point Triangulation**: Reconstructing the 3D coordinates of the scene points.
- **Reprojection Error**: Measuring the accuracy of the 3D reconstruction by calculating the reprojection error.

## Project Structure

- **estimate_trajectory.py**: Core script to estimate the camera trajectories based on the input images.
- **generate_point_cloud.py**: Script to generate a 3D point cloud from the estimated camera trajectories.
- **run.py**: Script to test and validate the implementation against provided test datasets.
- **common/**: Contains helper modules for dataset handling, intrinsics management, and trajectory processing.
- **tests/**: Includes the test datasets and ground truth for validating the SLAM implementation.

## Implementation Details

1. **Feature Detection and Description**: ORB is used to detect and describe key points in each image.
2. **Feature Matching**: Pairs of key points between images are matched, followed by filtering using the RANSAC algorithm.
3. **3D Reconstruction**: Triangulation is performed on the matched points to estimate the 3D coordinates of the scene points.
4. **Camera Pose Estimation**: The pose of the cameras for which positions are unknown is estimated by minimizing the reprojection error.

## How to Run

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Run the SLAM pipeline:
    ```bash
    python estimate_trajectory.py --input_path <path_to_input_data> --output_path <path_to_save_results>
    ```
3. Generate the 3D point cloud:
    ```bash
    python generate_point_cloud.py --trajectory_file <path_to_all_poses.txt> --output_path <path_to_save_point_cloud>
    ```

## Evaluation

The solution is evaluated based on the accuracy of the estimated camera positions compared to the ground truth. The primary metric is the percentage of camera positions estimated with an error of less than 20cm in translation and 15 degrees in rotation.

## References

- [OpenCV Documentation](https://docs.opencv.org)
- [TU Munich RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)

## Acknowledgements

This project was completed as part of the Computer Vision and Image Processing course at the Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University. Special thanks to the course instructors and the Graphics & Media Lab, Vision Group, for their guidance.

