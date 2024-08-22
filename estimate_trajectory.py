from common.dataset import Dataset
from common.trajectory import Trajectory
import cv2 as cv
import numpy as np
from itertools import combinations
from os.path import join
from scipy.spatial.distance import cdist


def compute_keypoints(img):
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def match_keypoints(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches


def matches_filtering(matches, kp1, kp2):
    pts1 = []
    pts2 = []

    for m in matches:
        if m.distance < 25:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def compute_fundamental(pts1, pts2):
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    return F, mask


def select_inliers(pts, mask):
    return pts[mask.ravel() == 1]


def get_all_supported_inliers(supported_imgs, data_dir):
    inliers = {}

    for pair in combinations(supported_imgs, 2):
        img1 = cv.imread(join(data_dir, pair[0][1]))
        img2 = cv.imread(join(data_dir, pair[1][1]))

        kp1, des1 = compute_keypoints(img1)
        kp2, des2 = compute_keypoints(img2)

        matches = match_keypoints(des1, des2)
        pts1, pts2 = matches_filtering(matches, kp1, kp2)
        if pts1.shape[0] < 8:
            continue
        F, mask = compute_fundamental(pts1, pts2)
        pts1 = select_inliers(pts1, mask)
        pts2 = select_inliers(pts2, mask)

        inliers[(int(pair[0][0]), int(pair[1][0]))] = np.stack((pts1, pts2))

    return inliers


def get_tracks(inliers):
    def get_track():
        last_img = imgs[-1]
        last_point = track[-1]

        if last_img >= 49:
            return None

        best_img = -1
        point_id = -1
        min_dist = 10000

        dist = lambda point: np.sqrt(np.square(point - last_point).sum(axis=1))

        for cur_img in range(last_img, 50):
            if (last_img, cur_img) not in pairs:
                continue

            points = inliers[(last_img, cur_img)][0]
            distances = dist(points)
            cur_min_dist = np.min(distances)

            if cur_min_dist < min_dist and cur_min_dist < 5:
                best_img = cur_img
                point_id = np.argmin(distances)

        if (last_img, best_img, point_id) in already_used or best_img == -1:
            return None

        points = inliers[(last_img, best_img)][1]

        track.append(np.array(points[point_id]))
        imgs.append(best_img)

        already_used.append((last_img, best_img, point_id))

        get_track()

    tracks = []
    already_used = []
    pairs = inliers.keys()

    for pair in pairs:
        points = inliers[pair]
        for point_id in range(points.shape[1]):
            if pair + (point_id,) in already_used:
                continue

            track = [np.array(points[0, point_id]), np.array(points[1, point_id])]
            imgs = list(pair)
            get_track()
            already_used.append(pair + (point_id,))

            if len(track) > 2:
                tracks.append([track, imgs])

    return tracks


def quaternion_to_rotation_matrix(quaternion):
    """
        Generate rotation matrix 3x3  from the unit quaternion.
        Input:
        qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
             (qx,qy,qz,qw) is the unit quaternion.

        Output:
        matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(((1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
                     (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
                     (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])), dtype=np.float64)


def get_all_matrixes(data_dir, n=50):
    rotations = np.empty((n, 3, 3))
    translations = np.empty((n, 3, 1))
    rodrigues = np.empty((n, 3, 1))
    projections = np.empty((n, 3, 4))

    filename = join(data_dir, 'intrinsics.txt')
    file = open(filename)
    for line in file:
        if '#' in line:
            continue
        fx, fy, cx, cy = list(map(float, line.split()))
        intrinsic = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

    filename = join(data_dir, 'known_poses.txt')
    file = open(filename)
    for line in file:
        if '#' in line:
            continue
        frame_id, tx, ty, tz, qx, qy, qz, qw = list(map(float, line.split()))
        frame_id = int(frame_id)
        translation = np.array([tx, ty, tz], ndmin=2)

        rotations[frame_id] = np.linalg.inv(quaternion_to_rotation_matrix((qx, qy, qz, qw)))
        rodrigues[frame_id], _ = cv.Rodrigues(rotations[frame_id])
        translations[frame_id] = -1.0 * np.matmul(rotations[frame_id], translation.T)
        projections[frame_id] = np.matmul(intrinsic,
                                          np.concatenate((rotations[frame_id], translations[frame_id]), axis=1))

    return intrinsic, rodrigues, translations, projections, rotations


def compute_3d_points(tracks, intrinsic, rodrigues, translations, projections):
    known_inliers = {}


    for track in tracks:
        temp_points = []
        good_track = True
        points, frames_id = track
        pairs = list(combinations(range(len(points)), 2))
        for i,j in pairs:
            cur_point = cv.triangulatePoints(projections[frames_id[i]], 
                                             projections[frames_id[j]],
                                             np.float_(points[i].reshape(1,-1).T), 
                                             np.float_(points[j].reshape(1,-1).T))
            if cur_point[3] != 0:
                cur_point = cur_point / cur_point[3]
                temp_points.append(cur_point)

        if len(temp_points) > 0:
            point_3d = np.array(temp_points).mean(axis=0)

            for i,id in enumerate(frames_id):
                point_reprojection, _ = cv.projectPoints(point_3d[0:3].T, 
                                                         rodrigues[id], 
                                                         translations[id], 
                                                         intrinsic, None)

                reprojection_error = np.linalg.norm(points[i].reshape(1,-1) - point_reprojection)

                if reprojection_error > 10:
                    good_track = False
                    break

        else:
            good_track = False

        if good_track:
            for i,frame_id in enumerate(frames_id):
                if known_inliers.get(frame_id):
                    known_inliers[frame_id][0].append(points[i])
                    known_inliers[frame_id][1].append(point_3d)
                else:
                    known_inliers[frame_id] = [[points[i]], [point_3d]]

    return known_inliers


def extract_points(tracks, bad_tracks):
    known_inliers = {}

    j = 0
    for track in tracks:
        if j in bad_tracks:
            j += 1
            continue
        point_3d = track[2]
        for i, frame_id in enumerate(track[1]):
            if known_inliers.get(frame_id):
                known_inliers[frame_id][0].append(track[0][i])
                known_inliers[frame_id][1].append(point_3d)
            else:
                known_inliers[frame_id] = [[track[0][i]], [point_3d]]
        j += 1
    return known_inliers


def get_all_unknown_inliers(supported_imgs, unknown_imgs, data_dir):
    unknown_inliers = {}

    for unknown_img in unknown_imgs:
        for supported_img in supported_imgs:
            img1 = cv.imread(join(data_dir, unknown_img[1]))
            img2 = cv.imread(join(data_dir, supported_img[1]))

            kp1, des1 = compute_keypoints(img1)
            kp2, des2 = compute_keypoints(img2)

            matches = match_keypoints(des1, des2)
            pts1, pts2 = matches_filtering(matches, kp1, kp2)
            if pts1.shape[0] < 8:
                continue
            F, mask = compute_fundamental(pts1, pts2)
            pts1 = select_inliers(pts1, mask)
            pts2 = select_inliers(pts2, mask)
            unknown_inliers[(int(unknown_img[0]), int(supported_img[0]))] = np.stack((pts1, pts2))

    return unknown_inliers


def match_2d_and_3d_points(unknown_inliers, known_inliers):
    points = {}
    
    for pair in unknown_inliers.keys():
        cont = False
        try:
            id_unknown = pair[0]
            id_known = pair[1]

            kps_unknown = unknown_inliers[pair][0]
            kps_known = unknown_inliers[pair][1]

            kps_with_3d = known_inliers[id_known][0]
            points_3d = known_inliers[id_known][1]

            if points.get(id_unknown) == None:
                points[id_unknown] = [[],[]]
        except:
            cont = True
            
        if cont:
            continue
        
        if points.get(id_unknown) == None:
            points[id_unknown] = [[],[]]
        
        for i,kp_known in enumerate(kps_known):
            for j, kp_3d in enumerate(kps_with_3d):
                if np.all(kp_known == kp_3d):
                    points[id_unknown][0].append(kps_unknown[i])
                    points[id_unknown][1].append(points_3d[j][:3])

    return points


def get_transformation_params(matches, intrinsic):
    results = {}

    for id in matches.keys():
        points_2d = np.ascontiguousarray(matches[id][0], np.float32).reshape(-1, 2)
        points_3d = np.ascontiguousarray(matches[id][1], np.float32).reshape(-1, 3)

        if points_2d.shape[0] < 5:
            continue

        retval, rvec, tvec, _ = cv.solvePnPRansac(points_3d, points_2d, intrinsic, None)
        if retval:
            temp = cv.Rodrigues(rvec)[0]
            trace = temp[0, 0] + temp[1, 1] + temp[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (temp[2, 1] - temp[1, 2]) * s
                qy = (temp[0, 2] - temp[2, 0]) * s
                qz = (temp[1, 0] - temp[0, 1]) * s
            elif temp[0, 0] > temp[1, 1] and temp[0, 0] > temp[2, 2]:
                s = 2.0 * np.sqrt(1.0 + temp[0, 0] - temp[1, 1] - temp[2, 2])
                qw = (temp[2, 1] - temp[1, 2]) / s
                qx = 0.25 * s
                qy = (temp[0, 1] + temp[1, 0]) / s
                qz = (temp[0, 2] + temp[2, 0]) / s
            elif temp[1, 1] > temp[2, 2]:
                s = 2.0 * np.sqrt(1.0 + temp[1, 1] - temp[0, 0] - temp[2, 2])
                qw = (temp[0, 2] - temp[2, 0]) / s
                qx = (temp[0, 1] + temp[1, 0]) / s
                qy = 0.25 * s
                qz = (temp[1, 2] + temp[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + temp[2, 2] - temp[0, 0] - temp[1, 1])
                qw = (temp[1, 0] - temp[0, 1]) / s
                qx = (temp[0, 2] + temp[2, 0]) / s
                qy = (temp[1, 2] + temp[2, 1]) / s
                qz = 0.25 * s

            results[id] = list(np.concatenate((-np.dot(tvec.ravel(), temp), np.array([qx, qy, qz, -qw]))))
    return results


def get_known_poses(data_dir):
    filename = join(data_dir, 'known_poses.txt')
    results = {}
    file = open(filename)
    for line in file:
        if '#' in line:
            continue
        frame_id, tx, ty, tz, qx, qy, qz, qw = list(map(float, line.split()))
        frame_id = int(frame_id)
        results[frame_id] = [tx, ty, tz, qx, qy, qz, qw]

    return results


def estimate_trajectory(data_dir, out_dir):
    files = open(join(data_dir, 'rgb.txt'))
    lines = files.readlines()
    files.close()

    files = list(map(lambda x: x.split(), lines))

    supported_imgs = files[:50]
    unknown_imgs = files[50:]

    inliers = get_all_supported_inliers(supported_imgs, data_dir)
    tracks = get_tracks(inliers)

    intrinsic, rodrigues, translations, projections, rotations = get_all_matrixes(data_dir)

    known_inliers = compute_3d_points(tracks, intrinsic, rodrigues, translations, projections)
    unknown_inliers = get_all_unknown_inliers(supported_imgs, unknown_imgs, data_dir)

    matches = match_2d_and_3d_points(unknown_inliers, known_inliers)

    results_1 = get_known_poses(data_dir)
    results_2 = get_transformation_params(matches, intrinsic)

    trajectory = {}

    for key in results_1.keys():
        trajectory[key] = results_1[key]
    for key in results_2.keys():
        trajectory[key] = results_2[key]

    with open(Dataset.get_result_poses_file(out_dir), 'w') as file:
        file.write(Trajectory.FILE_HEADER)

        for frame_id, pose in trajectory.items():
            file.write('{} {}\n'.format(frame_id, ' '.join(map(str, pose))))


