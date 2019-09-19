"""
TF records help methods.

Author: Daniel Koguciuk
"""

from __future__ import print_function

import os
import sys
import cv2
import pcl
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from sklearn.metrics.pairwise import pairwise_distances

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import hha_utils


def get_greedy_perm(distances):
    """
    A Naive O(mat_size^2) algorithm to do furthest points sampling

    Parameters
    ----------
    distances : ndarray (mat_size, mat_size)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (mat_size-length array of indices),
        lambdas (mat_size-length array of insertion radii))
    """

    mat_size = distances.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(mat_size, dtype=np.int64)
    lambdas = np.zeros(mat_size)
    ds = distances[0, :]
    for i in range(1, mat_size):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, distances[idx, :])
    return perm, lambdas


def prepare_data_pad(pcd_filepath, img_filepath, y_name, y_int):
    # Load and pad point cloud
    max_point_cloud_size = 44214
    point_cloud = pcl.load(pcd_filepath.decode('utf-8'))
    point_cloud = np.asarray(point_cloud)
    pad_size = max_point_cloud_size - len(point_cloud)
    point_cloud = np.pad(point_cloud, ((0, pad_size), (0, 0)), 'edge')
    return np.asarray(point_cloud), img_filepath, y_name, y_int


def prepare_data_fps(pcd_filepath, img_filepath, y_name, y_int, point_cloud_size=1024):
    # Load and pad point cloud
    point_cloud = pcl.load(pcd_filepath.decode('utf-8'))
    point_cloud = np.asarray(point_cloud)
    if len(point_cloud) > point_cloud_size:
        distances = pairwise_distances(point_cloud, metric='euclidean')
        (perm, _) = get_greedy_perm(distances)
        point_cloud = point_cloud[perm[:point_cloud_size]]
    else:
        pad_size = point_cloud_size - len(point_cloud)
        point_cloud = np.pad(point_cloud, ((0, pad_size), (0, 0)), 'edge')
    return np.asarray(point_cloud), img_filepath, y_name, y_int


def prepare_data_rnd(pcd_filepath, img_filepath, loc_filepath, y_name, y_int, point_cloud_size=1024):
    # Load and pad point cloud
    point_cloud = pcl.load(pcd_filepath.decode('utf-8'))
    point_cloud = np.asarray(point_cloud)
    if len(point_cloud) > point_cloud_size:
        indices = np.random.choice(len(point_cloud), point_cloud_size)
        point_cloud = point_cloud[indices]
    elif len(point_cloud) < point_cloud_size:
        pad_size = point_cloud_size - len(point_cloud)
        point_cloud = np.pad(point_cloud, ((0, pad_size), (0, 0)), 'edge')
    return np.asarray(point_cloud), img_filepath, loc_filepath, y_name, y_int


def create_point_cloud(depth_image, crop_top_left_corner=None, color_image=None):
    # Depth conversion
    depth = depth_image.astype(np.float32)
    depth[depth == 0] = np.nan
    # RGB-D camera constants
    center = [320, 240]
    image_h, image_w = depth.shape
    scale_f = 570.3
    mm_per_m = 1000
    if crop_top_left_corner is None:
        crop_top_left_corner = [0, 0]
    # Convert depth image to 3d point cloud
    channels = 3
    if color_image is not None:
        channels = 6
    point_cloud = np.zeros((image_h, image_w, channels), dtype=np.float32)
    x_grid = np.ones((image_h, 1), dtype=np.float32) * np.arange(image_w) + crop_top_left_corner[0] - center[0]
    y_grid = (np.arange(image_h).reshape(image_h, 1)*np.ones((1, image_w), dtype=np.float32) +
              crop_top_left_corner[1] - center[1])
    point_cloud[..., 0] = np.multiply(x_grid, depth) / scale_f / mm_per_m
    point_cloud[..., 1] = np.multiply(y_grid, depth) / scale_f / mm_per_m
    point_cloud[..., 2] = depth / mm_per_m
    # Assign color to point cloud
    if color_image is not None:
        point_cloud[..., 3:] = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    # Cut off to far points & nan handling
    point_cloud[point_cloud[..., 2] > 2.5] = np.nan
#    point_cloud = np.nan_to_num(point_cloud)
    return point_cloud.reshape(-1, 3)


def load_depth_and_create_point_cloud_data_rnd(pcd_filepath, img_filepath, loc_filepath, y_name, y_int,
                                               point_cloud_size=1024):
    # Load loc
    with open(loc_filepath, 'r') as f:
        loc = [int(l) for l in f.read().strip().split(',')]
    # Load depth and convert to float32
    depth = cv2.imread(pcd_filepath.decode('utf-8'), cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32)
    # Create point cloud
    point_cloud = create_point_cloud(depth, loc)
    # Erase rows with NaNs
    point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1)]
    # Sample point cloud
    if len(point_cloud) > point_cloud_size:
        indices = np.random.choice(len(point_cloud), point_cloud_size)
        point_cloud = point_cloud[indices]
    elif len(point_cloud) < point_cloud_size:
        pad_size = point_cloud_size - len(point_cloud)
        point_cloud = np.pad(point_cloud, ((0, pad_size), (0, 0)), 'edge')
    # Return
    return point_cloud, img_filepath, loc_filepath, y_name, y_int


def load_depth(pcd_filepath, img_filepath, loc_filepath, y_name, y_int, depth_size, scale=1.0):
    # Load loc
    with open(loc_filepath, 'r') as f:
        loc = [int(l) for l in f.read().strip().split(',')]
    # Load depth and convert to float32
    depth = cv2.imread(pcd_filepath.decode('utf-8'), cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32)
    # Reshape
    if depth.shape[0] < depth_size:
        pad_size = depth_size - depth.shape[0]
        pad_1 = np.random.choice(pad_size)
        pad_2 = pad_size - pad_1
        depth = np.pad(depth, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0.0, 0.0))
    if depth.shape[0] > depth_size:
        pad_size = depth.shape[0] - depth_size
        pad_1 = np.random.choice(pad_size)
        pad_2 = pad_size - pad_1
        depth = depth[pad_1:-pad_2]
    if depth.shape[1] < depth_size:
        pad_size = depth_size - depth.shape[1]
        pad_1 = np.random.choice(pad_size)
        pad_2 = pad_size - pad_1
        depth = np.pad(depth, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0.0, 0.0))
    if depth.shape[1] > depth_size:
        pad_size = depth.shape[1] - depth_size
        pad_1 = np.random.choice(pad_size)
        pad_2 = pad_size - pad_1
        depth = depth[:, pad_1:-pad_2]
    # Replace nans with 0
    # depth = np.nan_to_num(depth)
    if np.isnan(depth).any():
        print('NaN in input {}'.format(pcd_filepath))
        exit()
    # Scale depth channel
    depth *= scale
    # Return
    return depth, img_filepath, loc_filepath, y_name, y_int


def load_depth_in_hha(pcd_filepath, img_filepath, loc_filepath, y_name, y_int, depth_size):
    # Load loc
    with open(loc_filepath, 'r') as f:
        loc = [int(l) for l in f.read().strip().split(',')]
    # Load depth
    depth_whole = np.zeros((480, 640), dtype=np.float32)
    depth_object = cv2.imread(pcd_filepath.decode('utf-8'), cv2.IMREAD_ANYDEPTH) / 1000
    depth_whole[loc[1]-1:loc[1]-1+depth_object.shape[0], loc[0]-1:loc[0]-1+depth_object.shape[1]] = depth_object

    # Calc HHA representation
    cam_params = hha_utils.get_camera_param_washington()
    hha_image = hha_utils.get_HHA(cam_params, depth_whole, depth_whole)
    hha_image = hha_image[loc[1]-1:loc[1]-1+depth_object.shape[0], loc[0]-1:loc[0]-1+depth_object.shape[1]]
    hha_image = hha_utils.resize_ratio_padding(hha_image, depth_size)

    # Return
    return hha_image.astype(np.float32), img_filepath, loc_filepath, y_name, y_int


def augment_depth_image(depth_image, img_filepath, loc_filepath, y_name, y_int):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25, 25)),
        iaa.GaussianBlur(sigma=(0, 1)),
        iaa.Fliplr(p=0.5),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    ], random_order=True)
    depth_image = seq.augment_image(depth_image)
    # Return
    return depth_image, img_filepath, loc_filepath, y_name, y_int


def _shuffle_points(point_cloud):
    """
    Shuffle points in point cloud and return its random permutation.
    Args:
        point_cloud (numpy.ndarray of size [N,3]): point clouds data to be shuffled along first axis
    Returns:
        (numpy.ndarray of size [N,3]): shuffled point cloud
    """
    idx = np.arange(point_cloud.shape[0])
    np.random.shuffle(idx)
    return point_cloud[idx, ...]


def _subtract_the_mean(point_cloud):
    """
    Subtract the mean in point cloud and return its zero-mean version.
    Args:
        point_cloud (numpy.ndarray of size [N,3]): point cloud
    Returns:
        (numpy.ndarray of size [N,3]): point cloud with zero-mean
    """
    point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    return point_cloud


def _jitter_points(point_cloud, sigma=0.001, clip=0.005):
    """
    Randomly jitter points

    Args:
        point_cloud (np.ndarray of size [N, 3]): Point cloud
        sigma (float): Sigma value of gaussian noise to be applied pointwise.
        clip (float): Clipping value of gaussian noise.
    Returns:
          (np.ndarray of size [N, 3]): Jittered point cloud data.
    """
    # Get size
    n, c = point_cloud.shape

    # Generate noise
    if clip <= 0:
        raise ValueError("Clip should be a positive number")
    jittered_data = np.clip(sigma * np.random.randn(n, c), -1 * clip, clip)

    # Add to pointcloud
    point_cloud += jittered_data
    return point_cloud


def augment_point_cloud(point_cloud, img_filepath, loc_filepath, y_name, y_int,
                        permute_points=True, jitter_points=True, subtract_the_mean=True):
    # Permute points
    if permute_points:
        point_cloud = _shuffle_points(point_cloud)
    # Jitter points
    if jitter_points:
        point_cloud = _jitter_points(point_cloud)
    # Jitter points
    if subtract_the_mean:
        point_cloud = _subtract_the_mean(point_cloud)
    # Return
    return point_cloud, img_filepath, loc_filepath, y_name, y_int


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_string(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def wrap_bytes(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrap_floats(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def point_cloud_data_to_tfexample(pc_data, label_name, label_int):

    data = {'point-cloud/ravel': wrap_floats(pc_data.ravel()),
            'point-cloud/shape0': wrap_int64(pc_data.shape[0]),
            'point-cloud/shape1': wrap_int64(pc_data.shape[1]),
            'label/name': wrap_string(label_name.encode('utf-8')),
            'label/int': wrap_int64(label_int),
            }
    return tf.train.Example(features=tf.train.Features(feature=data))


def paths_to_tfexample(pc_path, image_path, loc_path, label_name, label_int):

    data = {'data/pcd_path': wrap_string(pc_path.encode('utf-8')),
            'data/img_path': wrap_string(image_path.encode('utf-8')),
            'data/loc_path': wrap_string(loc_path.encode('utf-8')),
            'label/name': wrap_string(label_name.encode('utf-8')),
            'label/int': wrap_int64(label_int),
            }
    return tf.train.Example(features=tf.train.Features(feature=data))


def tfexample_to_paths(example_proto):
    # Parse single example
    features = {'data/pcd_path':  tf.FixedLenFeature((), tf.string),
                'data/img_path':  tf.FixedLenFeature((), tf.string),
                'data/loc_path': tf.FixedLenFeature((), tf.string),
                'label/name': tf.FixedLenFeature((), tf.string),
                'label/int': tf.FixedLenFeature((), tf.int64),
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    # Reshape data
    x = {'pcd_path': parsed_features['data/pcd_path'],
         'img_path': parsed_features['data/img_path'],
         'loc_path': parsed_features['data/loc_path']}
    y = {'name': parsed_features['label/name'],
         'int': parsed_features['label/int']}

    # Return
    return x, y


def tfexample_to_point_cloud(example_proto):
    # Parse single example
    features = {'point-cloud/ravel': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'point-cloud/shape0': tf.FixedLenFeature((), tf.int64),
                'point-cloud/shape1': tf.FixedLenFeature((), tf.int64),
                'label/name': tf.FixedLenFeature((), tf.string),
                'label/int': tf.FixedLenFeature((), tf.int64),
                }
    parsed_features = tf.parse_single_example(example_proto, features)
    # Reshape data
    x = tf.reshape(parsed_features['point-cloud/ravel'], [parsed_features['point-cloud/shape0'],
                                                          parsed_features['point-cloud/shape1']])
    y = {'name': parsed_features['label/name'],
         'int': parsed_features['label/int']}

    # Return
    return x, y
