import glob
import os
import numpy as np
import cv2
import click
import json
from json import JSONEncoder


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class Calibrator:
    def __init__(self, imageSize, cb_shape, cb_size):
        self.cb_shape = tuple([int(x) for x in cb_shape.split('x')])
        self.pattern_points = np.zeros((np.prod(self.cb_shape), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.cb_shape).T.reshape(-1, 2)
        self.pattern_points *= cb_size

        self.imageSize = tuple([int(x) for x in imageSize.split('x')])
        self.alpha = -1

        self.term = (cv2.TERM_CRITERIA_EPS +
                     cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.arrays = None
        self.calibration = None

    def read_images(self, dir):
        assert os.path.isdir(dir+'/left') and os.path.isdir(dir+'/right')

        def find_corners(p):
            img = cv2.imread(p, 0)
            img = cv2.resize(img, self.imageSize)
            ret, corners = cv2.findChessboardCorners(
                img, self.cb_shape, cv2.CALIB_CB_FAST_CHECK)
            if ret and img.shape[::-1] == self.imageSize:
                cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.term)
                return [os.path.basename(p), self.pattern_points, corners]

        arr_left = np.array([find_corners(p)
                             for p in sorted(glob.glob(f"{dir}/left/*.png"))])
        arr_left = arr_left[arr_left != None][0]

        arr_right = np.array([find_corners(p)
                              for p in sorted(glob.glob(f"{dir}/right/*.png"))])
        arr_right = arr_right[arr_right != None][0]

        all_names = sorted(list(set(arr_left[:, 0]) & set(arr_right[:, 0])))

        def get_intersection(arr, all_names):
            return arr[np.isin(arr[:, 0], all_names)]

        arr_left = get_intersection(arr_left, all_names)
        arr_right = get_intersection(arr_right, all_names)

        self.arrays = [arr_left, arr_right]
        print(f'Found {len(arr_left)} images with chessboard')

    def calibrate_cameras(self):
        assert self.arrays
        matrix_left, distortion_left = cv2.calibrateCamera(
            self.arrays[0][:, 1], self.arrays[0][:, 2], self.imageSize, None, None)[1:3]
        matrix_right, distortion_right = cv2.calibrateCamera(
            self.arrays[0][:, 1], self.arrays[1][:, 2], self.imageSize, None, None)[1:3]

        rot_matrix, trans_vector = cv2.stereoCalibrate(
            self.arrays[0][:, 1], self.arrays[0][:, 2], self.arrays[1][:, 2],
            matrix_left, distortion_left,
            matrix_right, distortion_right,
            self.imageSize, flags=cv2.CALIB_FIX_INTRINSIC, criteria=self.term)[5:7]

        rect_left, rect_right, proj_left, proj_right, dispartity, ROI_left, ROI_right = cv2.stereoRectify(
            matrix_left, distortion_left,
            matrix_right, distortion_right,
            self.imageSize, rot_matrix, trans_vector,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=self.alpha)

        self.calibration = {
            'general': {
                'rotation': rot_matrix,
                'translation': trans_vector,
                'dispartity': dispartity,
            },
            'left': {
                'matrix': matrix_left,
                'distortion': distortion_left,
                'rectification': rect_left,
                'projection': proj_left,
                'ROI': ROI_left,
            },
            'right': {
                'matrix': matrix_right,
                'distortion': distortion_right,
                'rectification': rect_right,
                'projection': proj_right,
                'ROI': ROI_right,
            }
        }

    def save(self, path):
        assert self.calibration

        with open(path, 'w') as f:
            json.dump(self.calibration, f, cls=NumpyArrayEncoder)


@click.command()
@click.option('--dir', default='imgs', help='Source directory', required=True)
@click.option('--dest', default='params.json', help='Destination json file', required=True)
@click.option('--size', default='1270x720', help='Image size', required=True)
@click.option('--cb-shape', default='7x6', help='Chessboard size (COLSxROWS)', required=True)
@click.option('--cb-size', default=0.0417, help='Size of one chessboard square [m]', required=True)
def main(dir, dest, size, cb_shape, cb_size):
    calibr = Calibrator(size, cb_shape, cb_size)
    calibr.read_images(dir)
    calibr.calibrate_cameras()
    calibr.save(dest)


if __name__ == "__main__":
    main()
