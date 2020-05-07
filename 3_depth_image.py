import glob
import os
import numpy as np
import cv2
import click
import json
from json import JSONEncoder


class Depther:
    def __init__(self, source, size):
        self.size = tuple([int(x) for x in size.split('x')])

        with open(source, 'r') as f:
            self.calibration = json.load(f)

        self.map_left = cv2.initUndistortRectifyMap(
            np.array(self.calibration['left']['matrix']),
            np.array(self.calibration['left']['distortion']),
            np.array(self.calibration['left']['rectification']),
            np.array(self.calibration['left']['projection']),
            self.size, cv2.CV_32FC1)

        self.map_right = cv2.initUndistortRectifyMap(
            np.array(self.calibration['right']['matrix']),
            np.array(self.calibration['right']['distortion']),
            np.array(self.calibration['right']['rectification']),
            np.array(self.calibration['right']['projection']),
            self.size, cv2.CV_32FC1)

        self.stereo = cv2.StereoSGBM_create(
            minDisparity = 1,
            numDisparities = 128,
            blockSize = 3,
            P1 = 8*3**2,
            P2 = 32*3**2,
            disp12MaxDiff = 1,
            preFilterCap = 63,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            mode = 0
        )


    def make_image(self, left, right):
        remaped_left = cv2.remap(left, self.map_left[0], self.map_left[1], cv2.INTER_LINEAR)
        remaped_right = cv2.remap(right, self.map_right[0], self.map_right[1], cv2.INTER_LINEAR)

        grey_left = cv2.cvtColor(remaped_left, cv2.COLOR_BGR2GRAY)
        grey_right = cv2.cvtColor(remaped_right, cv2.COLOR_BGR2GRAY)
        depth = self.stereo.compute(grey_left, grey_right)

        depth = depth*255. / (depth.max()-depth.min())
        depth = depth.astype(np.uint8)

        depth = cv2.medianBlur(depth,9)        

        return remaped_left, remaped_right, depth

    def compute_images(self, imgl, imgr):
        left = cv2.imread(imgl)
        left = cv2.resize(left, self.size)
        right = cv2.imread(imgr)
        right = cv2.resize(right, self.size)

        remaped_left, remaped_right, depth = self.make_image(left, right)

        cv2.imshow('orginal', cv2.resize(
            np.hstack([left, right]), None, fx=0.6, fy=0.6))
        cv2.imshow('remaped', cv2.resize(
            np.hstack([remaped_left, remaped_right]), None, fx=0.6, fy=0.6))
        cv2.imshow('depth', cv2.resize(depth, None, fx=0.6, fy=0.6))

        cv2.waitKey()
        cv2.destroyAllWindows()

    def compute_captures(self, capl, capr, flip=True):
        cap_left = cv2.VideoCapture(capl)
        cap_right = cv2.VideoCapture(capr)

        while(True):
            if not (cap_left.grab() and cap_right.grab()):
                break

            _, frame_left = cap_left.retrieve()
            _, frame_right = cap_right.retrieve()

            if flip:
                frame_left = cv2.flip(frame_left, -1)
                frame_right = cv2.flip(frame_right, -1)
                
            remaped_left, remaped_right, depth = self.make_image(frame_left, frame_right)
            cv2.imshow('orginal', cv2.resize(
                np.hstack([frame_left, frame_right]), None, fx=0.4, fy=0.4))
            cv2.imshow('remaped', cv2.resize(
                np.hstack([remaped_left, remaped_right]), None, fx=0.4, fy=0.4))
            cv2.imshow('depth', cv2.resize(depth, None, fx=0.4, fy=0.4))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()



@click.command()
@click.option('--source', default='params.json', help='Destination json file', required=True)
@click.option('--size', default='1280x720', help='Image size', required=True)
@click.option('--imgl', default='samples/left/000000.png', help='Left camera image')
@click.option('--imgr', default='samples/right/000000.png', help='Right camera image')
@click.option('--capl', default=-1, help='Video capture index (left)')
@click.option('--capr', default=-1, help='Video capture index (right)')
def main(source, size, imgl, imgr, capl, capr):
    dep = Depther(source, size)

    if capl>=0 and capr>=0:
        dep.compute_captures(capl, capr)
    else:
        dep.compute_images(imgl, imgr)


if __name__ == "__main__":
    main()
