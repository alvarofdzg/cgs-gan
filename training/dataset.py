# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import re
# from utils.io import read_file

import cv2
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pyspng


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            all_labels = np.array([label for label in self._raw_labels.values()])
            self._raw_labels_std = all_labels.std(0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        label = self._get_raw_labels()[fname.split('.')[0] + ".png"]
        label = np.array(label)
        if self._xflip[idx] == 1:
            flipped_pose = flip_yaw(label[:16].reshape(4, 4)).reshape(-1)
            label[:16] = flipped_pose
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        fname = self._image_fnames[self._raw_idx[idx]]
        d.raw_label = self._get_raw_labels()[fname.split('.')[0] + ".png"].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels[list(raw_labels.keys())[0]].shape
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        path,
        resolution=None,
        camera_sample_mode=None,
        rand_background=True,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self.mask_path = os.path.join(os.path.dirname(path), "mask")
        self.rand_background = rand_background

        print(f"using {camera_sample_mode} camera_sample_mode")
        self.camera_sample_mode = camera_sample_mode

        with open(os.path.join(f'./custom_dist/{camera_sample_mode}.json'), "r") as f:
            index_list = json.load(f)

        # original code looks through the direcotry
        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        # scan all images in folder
        available_indices = set([int(re.findall(r"\d+", fname)[0]) for fname in self._image_fnames])
        filtered_indices = [i for i in index_list if i in available_indices]

        print("Images in directory:", len(self._image_fnames))
        print("Oversampled Images:", len(filtered_indices))
        print("Unique Images:", len(set(filtered_indices)))

        file_ending = self._image_fnames[0].split(".")[-1]
        self._image_fnames = [f"{i:05d}.{file_ending}" for i in filtered_indices]

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
            if image.ndim == 2:
                image = image[:, :, np.newaxis]  # HW => HWC

        if self.mask_path is not None:
            mask_fname = os.path.join(self.mask_path, self._image_fnames[raw_idx].split(".")[0] + ".png")
            with self._open_file(mask_fname) as f:
                mask_image = pyspng.load(f.read())
            if mask_image.ndim == 2:
                mask_image = cv2.resize(mask_image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_LINEAR)
                mask_image = mask_image[:, :, np.newaxis]  # HW => HWC

            if self.rand_background:
                bg = np.ones_like(image)
                bg[..., 0] = np.random.randint(low=0, high=255)
                bg[..., 1] = np.random.randint(low=0, high=255)
                bg[..., 2] = np.random.randint(low=0, high=255)
            else:
                bg = np.ones_like(image) * 255
            image = ((mask_image / 255) * image + (1 - mask_image / 255) * bg).astype(np.uint8)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = os.path.join(os.path.dirname(self._path), "dataset.json") # dataset_recrop.json
        with self._open_file(fname) as f:
            print(f"loading labels from {fname}")
            cam_labels = json.load(f)["labels"]
            if cam_labels is None:
                return None
        cam_labels = dict(cam_labels)
        for key in cam_labels.keys():
            cam_labels[key] = np.array(cam_labels[key], dtype=np.float32)
        return cam_labels


class ImageFolderDataset2(Dataset):
    def __init__(
        self,
        path,
        resolution=None,
        camera_sample_mode=None,
        rand_background=True,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self.rand_background = rand_background

        self.mask_images = "/data1/datasets/dataset_alvaro/segmentation/synthesisai"
        self.theta = [theta for theta in range(0, 360, 30)]
        # self.theta = [theta for theta in range(0, 120, 30)] + [theta for theta in range(270, 360, 30)]
        # self.theta = [theta for theta in range(0, 90, 30)] + [theta for theta in range(300, 360, 30)]
        # self.theta = [0, 30, 60, 300, 330]
        # self.azimuth = [0]
        self.azimuth = [0, 25, 335]
        # self.expressions = ["neutral"]
        self.expressions = ["neutral", "happy", "scared"]
        # self.subjects_ids = [1]
        self.subjects_ids = [idx for idx in range(1, 301)]

        self._image_fnames = []
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        self._image_fnames.append(
                            os.path.join(
                                self._path,
                                "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.npy".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                            )
                        )
                        
        self._type = "dir"
        
        self.labels = self._load_raw_labels()
        all_labels = np.array([label for label in self.labels.values()])
        self._raw_labels_std = all_labels.std(0)

        PIL.Image.init()

        print("Images in directory:", len(self._image_fnames))

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")
         
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_raw_labels(self):
        return self.labels

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        label = self.labels[os.path.basename(fname).split('.')[0]]
        label = np.array(label)
        if self._xflip[idx] == 1:
            flipped_pose = flip_yaw(label[:16].reshape(4, 4)).reshape(-1)
            label[:16] = flipped_pose
        return label.copy()
    
    # def _load_raw_image(self, raw_idx):
    #     fname = self._image_fnames[raw_idx]
    #     mask_name = os.path.join(self.mask_images, os.path.basename(fname).split(".")[0] + ".npy")

    #     with self._open_file(fname) as f:
    #         image = np.load(f)
    #     with self._open_file(mask_name) as f:
    #         mask_image = np.load(f)
    #         mask_image = mask_image[:, :, np.newaxis]  # HW => HWC
    #     if self.rand_background:
    #         bg = np.ones_like(image)
    #         bg[..., 0] = np.random.randint(low=0, high=255)
    #         bg[..., 1] = np.random.randint(low=0, high=255)
    #         bg[..., 2] = np.random.randint(low=0, high=255)
    #     else:
    #         bg = np.ones_like(image) * 255
    #     image = (mask_image * image + (1 - mask_image) * bg).astype(np.uint8)
        
    #     image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    #     image = image.transpose(2, 0, 1)  # HWC => CHW
    #     return image
    
    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        mask_name = os.path.join(self.mask_images, os.path.basename(fname).split(".")[0] + ".npy")

        with self._open_file(fname) as f:
            image = np.load(f)
        with self._open_file(mask_name) as f:
            mask_image = np.load(f)
            
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = mask_image[:, :, np.newaxis]  # HW => HWC
        
        if self.rand_background:
            bg = np.ones_like(image)
            bg[..., 0] = np.random.randint(low=0, high=255)
            bg[..., 1] = np.random.randint(low=0, high=255)
            bg[..., 2] = np.random.randint(low=0, high=255)
        else:
            bg = np.ones_like(image) * 255
        image = (mask_image * image + (1 - mask_image) * bg).astype(np.uint8)
        
        # image = image[150:800, 187:-187]

        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image
    
    def _load_raw_labels(self):
        camera_labels = {}
        camera_path = "/data1/datasets/synthesis_ai/labels_alvaro"
        
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        
                        label_path = os.path.join(
                            camera_path,
                            "subject_{:01d}".format(cur_subject_id),
                            cur_expression,
                            "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.json".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                        )
                        with open(label_path, "r") as f:
                            cur_label = json.load(f)

                        cur_extrinsics = np.concatenate(
                            [
                                np.array(cur_label["M"], dtype=np.float32),
                                np.array([[0, 0, 0, 1]], dtype=np.float32),  # Add identity quaternion
                            ]
                        )
                        # Convert from World2Cam to Cam2World
                        cur_extrinsics = np.linalg.inv(cur_extrinsics)
                        
                        #Normalize intrinsics
                        intrinsics = np.array(cur_label["K"], dtype=np.float32)
                        intrinsics[:2] /= 1024

                        prepared_label = np.concatenate(
                            [
                                cur_extrinsics.flatten(),
                                intrinsics.flatten()
                            ]
                        )
                        camera_labels["subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta)
                        ] = prepared_label
        return camera_labels

"""
This dataset is for test_4
"""
class ImageFolderDataset3(Dataset):
    def __init__(
        self,
        path,
        resolution=None,
        camera_sample_mode=None,
        rand_background=True,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self.mask_path = os.path.join(os.path.dirname(path), "mask")
        self.rand_background = rand_background

        self.mask_images = "/data1/datasets/dataset_alvaro/segmentation/synthesisai"
        # self.mask_images = "/data1/datasets/dataset_alvaro/segmentation/synthesisai_face"
        # self.theta = [theta for theta in range(0, 360, 30)]
        # self.theta = [theta for theta in range(0, 120, 30)] + [theta for theta in range(270, 360, 30)]
        self.theta = [theta for theta in range(0, 90, 30)] + [theta for theta in range(300, 360, 30)]
        # self.theta = [0]
        self.azimuth = [0, 25, 335]
        # self.azimuth = [0]
        self.expressions = ["neutral"]
        # self.expressions = ["neutral", "happy", "scared"]
        self.subjects_ids = [1]
        # self.subjects_ids = [1, 2, 3, 4]
        # self.subjects_ids = [idx for idx in range(1, 301)]

        self._image_fnames = []
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        self._image_fnames.append(
                            os.path.join(
                                self._path,
                                "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.npy".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                            )
                        )
                        
        self._type = "dir"
        
        self.labels = self._load_raw_labels()
        all_labels = np.array([label for label in self.labels.values()])
        self._raw_labels_std = all_labels.std(0)

        PIL.Image.init()

        print("Images in directory:", len(self._image_fnames))

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")
         
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_raw_labels(self):
        return self.labels

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        label = self.labels[os.path.basename(fname).split('.')[0]]
        label = np.array(label)
        if self._xflip[idx] == 1:
            flipped_pose = flip_yaw(label[:16].reshape(4, 4)).reshape(-1)
            label[:16] = flipped_pose
        return label.copy()
    
    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        mask_name = os.path.join(self.mask_images, os.path.basename(fname).split(".")[0] + ".npy")

        with self._open_file(fname) as f:
            image = np.load(f)
        with self._open_file(mask_name) as f:
            mask_image = np.load(f)
            
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = mask_image[:, :, np.newaxis]  # HW => HWC
        
        if self.rand_background:
            bg = np.ones_like(image)
            bg[..., 0] = np.random.randint(low=0, high=255)
            bg[..., 1] = np.random.randint(low=0, high=255)
            bg[..., 2] = np.random.randint(low=0, high=255)
        else:
            bg = np.ones_like(image) * 255
        image = (mask_image * image + (1 - mask_image) * bg).astype(np.uint8)
        
        # image = image[150:800, 187:-187]

        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        camera_labels = {}
        camera_path = "/data1/datasets/synthesis_ai/labels_alvaro"
        
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        
                        label_path = os.path.join(
                            camera_path,
                            "subject_{:01d}".format(cur_subject_id),
                            cur_expression,
                            "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.json".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                        )
                        with open(label_path, "r") as f:
                            cur_label = json.load(f)

                        cur_extrinsics = np.concatenate(
                            [
                                np.array(cur_label["M"], dtype=np.float32),
                                np.array([[0, 0, 0, 1]], dtype=np.float32),  # Add identity quaternion
                            ]
                        )
                        # Convert from World2Cam to Cam2World
                        cur_extrinsics = np.linalg.inv(cur_extrinsics)
                        
                        #Normalize intrinsics
                        intrinsics = np.array(cur_label["K"], dtype=np.float32)
                        intrinsics[:2] /= 1024

                        prepared_label = np.concatenate(
                            [
                                cur_extrinsics.flatten(),
                                intrinsics.flatten()
                            ]
                        )
                        camera_labels["subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta)
                        ] = prepared_label
        return camera_labels

"""
This dataset is for test_5
"""
class ImageFolderDataset4(Dataset):
    def __init__(
        self,
        path,
        resolution=None,
        camera_sample_mode=None,
        rand_background=True,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None
        self.mask_path = os.path.join(os.path.dirname(path), "mask")
        self.rand_background = rand_background

        self.mask_images = "/data1/datasets/dataset_alvaro/segmentation/synthesisai_face"
        # self.theta = [theta for theta in range(0, 360, 30)]
        # self.theta = [theta for theta in range(0, 120, 30)] + [theta for theta in range(270, 360, 30)]
        # self.theta = [theta for theta in range(0, 90, 30)] + [theta for theta in range(300, 360, 30)]
        self.theta = [0]
        # self.azimuth = [0, 25, 335]
        self.azimuth = [0]
        self.expressions = ["neutral"]
        # self.expressions = ["neutral", "happy", "scared"]
        self.subjects_ids = [40]
        # self.subjects_ids = [1, 2, 3, 4]
        # self.subjects_ids = [idx for idx in range(1, 301)]

        self.x_0 = 212
        self.y_0 = 175
        self.x_t = 812
        self.y_t = 775

        self._image_fnames = []
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        self._image_fnames.append(
                            os.path.join(
                                self._path,
                                "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.npy".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                            )
                        )
                        
        self._type = "dir"
        
        # self.image = self._load_raw_image_predefined(raw_idx=0)
        
        self.labels = self._load_raw_labels()
        all_labels = np.array([label for label in self.labels.values()])
        self._raw_labels_std = all_labels.std(0)

        PIL.Image.init()

        print("Images in directory:", len(self._image_fnames))

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")
         
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        # Removing flip data augmentation
        self._xflip = np.zeros_like(self._xflip, dtype=np.uint8)
        

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_raw_labels(self):
        return self.labels

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        label = self.labels[os.path.basename(fname).split('.')[0]]
        label = np.array(label)
        if self._xflip[idx] == 1:
            flipped_pose = flip_yaw(label[:16].reshape(4, 4)).reshape(-1)
            label[:16] = flipped_pose
        return label.copy()
    
    # def _load_raw_image(self, raw_idx):
    #     return self.image
    
    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        mask_name = os.path.join(self.mask_images, os.path.basename(fname).split(".")[0] + ".npy")

        with self._open_file(fname) as f:
            image = np.load(f)
        with self._open_file(mask_name) as f:
            mask_image = np.load(f)
        
        image = image[self.y_0:self.y_t, self.x_0:self.x_t]
        mask_image = mask_image[self.y_0:self.y_t, self.x_0:self.x_t]
        
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_image = mask_image[:, :, np.newaxis]  # HW => HWC
        
        if self.rand_background:
            bg = np.ones_like(image)
            bg[..., 0] = np.random.randint(low=0, high=255)
            bg[..., 1] = np.random.randint(low=0, high=255)
            bg[..., 2] = np.random.randint(low=0, high=255)
        else:
            bg = np.ones_like(image) * 255
        image = (mask_image * image + (1 - mask_image) * bg).astype(np.uint8)
        
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        camera_labels = {}
        camera_path = "/data1/datasets/synthesis_ai/labels_alvaro"
        
        for cur_subject_id in self.subjects_ids:
            for cur_expression in self.expressions:
                for cur_azimuth in self.azimuth:
                    for cur_theta in self.theta:
                        
                        label_path = os.path.join(
                            camera_path,
                            "subject_{:01d}".format(cur_subject_id),
                            cur_expression,
                            "subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}.json".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta),
                        )
                        with open(label_path, "r") as f:
                            cur_label = json.load(f)

                        cur_extrinsics = np.concatenate(
                            [
                                np.array(cur_label["M"], dtype=np.float32),
                                np.array([[0, 0, 0, 1]], dtype=np.float32),  # Add identity quaternion
                            ]
                        )
                        # Convert from World2Cam to Cam2World
                        cur_extrinsics = np.linalg.inv(cur_extrinsics)
                        
                        #Normalize intrinsics
                        intrinsics = np.array(cur_label["K"], dtype=np.float32)
                        
                        intrinsics[0, 0] /= (self.x_t - self.x_0)
                        intrinsics[1, 1] /= (self.y_t - self.y_0)
                        
                        intrinsics[0, 2] = (intrinsics[0, 2] - self.x_0) / (self.x_t - self.x_0)
                        intrinsics[1, 2] = (intrinsics[1, 2] - self.y_0) / (self.y_t - self.y_0)
                        
                        prepared_label = np.concatenate(
                            [
                                cur_extrinsics.flatten(),
                                intrinsics.flatten()
                            ]
                        )
                        camera_labels["subject_{:01d}__{}__azimuth_{:01d}__theta_{:01d}".format(cur_subject_id, cur_expression, cur_azimuth, cur_theta)
                        ] = prepared_label
        return camera_labels
    
    
"""
This dataset is for Nersemble
"""
class ImageFolderDatasetNersemble(Dataset):
    def __init__(
        self,
        path,
        resolution=None,
        camera_sample_mode=None,
        rand_background=True,
        **super_kwargs,
    ):
        self._path = "/data/hs_generative_model_data/NeRSemble/data"
        self._zipfile = None
        self.mask_path = os.path.join(os.path.dirname(path), "mask")
        self.rand_background = rand_background

        self.subject_id = "187"

        # self._image_fnames = read_file(
        #     os.path.join(
        #         self._path,
        #         "nersemble_paths.txt"
        #     )
        # )
        self._image_fnames = [
            f"{self.subject_id}/extra_sequences/EMO-2-surprise+fear/frame_00006/images-2fps/cam_222200037.jpg"
        ]

        self.x_0 = 0
        self.y_0 = 350
        self.x_t = 2200
        self.y_t = 2908
        
        self._type = "dir"

        self.labels = self._load_raw_labels()
        all_labels = np.array([label for label in self.labels.values()])
        self._raw_labels_std = all_labels.std(0)

        PIL.Image.init()

        print("Images in directory:", len(self._image_fnames))

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")
         
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        # Removing flip data augmentation
        self._xflip = np.zeros_like(self._xflip, dtype=np.uint8)
        

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_raw_labels(self):
        return self.labels

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        label = self.labels[os.path.basename(fname).split('.')[0]]
        label = np.array(label)
        if self._xflip[idx] == 1:
            flipped_pose = flip_yaw(label[:16].reshape(4, 4)).reshape(-1)
            label[:16] = flipped_pose
        return label.copy()

    def _load_raw_image(self, raw_idx):
        fname = os.path.join(
            self._path,
            self._image_fnames[raw_idx]
        )
        # mask_name = os.path.join(self.mask_images, os.path.basename(fname).split(".")[0] + ".npy")

        image = np.array(read_file(file_path=fname))
        # with self._open_file(mask_name) as f:
        #     mask_image = np.load(f)
        
        image = image[self.y_0:self.y_t, self.x_0:self.x_t]
        # mask_image = mask_image[self.y_0:self.y_t, self.x_0:self.x_t]
        
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        # mask_image = cv2.resize(mask_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        # mask_image = mask_image[:, :, np.newaxis]  # HW => HWC
        
        # if self.rand_background:
        #     bg = np.ones_like(image)
        #     bg[..., 0] = np.random.randint(low=0, high=255)
        #     bg[..., 1] = np.random.randint(low=0, high=255)
        #     bg[..., 2] = np.random.randint(low=0, high=255)
        # else:
        #     bg = np.ones_like(image) * 255
        # image = (mask_image * image + (1 - mask_image) * bg).astype(np.uint8)
        
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        camera_path = os.path.join(
            self._path,
            f"{self.subject_id}/camera_params.json"
        )
        
        cur_label = read_file(
            file_path=camera_path
        )
        
        cur_extrinsics = np.array(cur_label["world_2_cam"]["222200037"], dtype=np.float32)
        # Convert from World2Cam to Cam2World
        cur_extrinsics = np.linalg.inv(cur_extrinsics)
        
        #Normalize intrinsics
        intrinsics = np.array(cur_label["intrinsics"], dtype=np.float32)
        intrinsics[:1] /= 2200
        intrinsics[1:2] /= 3208

        prepared_label = np.concatenate(
            [
                cur_extrinsics.flatten(),
                intrinsics.flatten()
            ]
        )
        camera_labels = {
            "cam_222200037":
                prepared_label
        }
        return camera_labels