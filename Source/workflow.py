# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the "Planetary System LRGB Aligner" tool (PSLA).
https://github.com/Rolf-Hempel/PlanetarySystemLrgbAligner

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSLA.  If not, see <http://www.gnu.org/licenses/>.

"""

import time

import cv2
import numpy as np
from PyQt5 import QtCore


class Workflow(QtCore.QThread):
    """
    The Workflow class creates a thread which runs in parallel to the main gui. Communication
    between the main gui and the workflow thread is realized as follows: Actions in this
    thread are triggered by flags set in the main gui thread. In the reverse direction, this
    thread emits signals which are connected with methods in the main gui.

    """

    # Define the list of signals with which this thread communicates with the main gui.
    set_status_signal = QtCore.pyqtSignal(int)

    def __init__(self, gui, parent=None):
        """
        Establish the connection with the main gui, set some instance variables and initialize all
        flags to False.

        :param gui: main gui object
        """

        QtCore.QThread.__init__(self, parent)
        self.gui = gui
        self.configuration = self.gui.configuration

        # Initialize some status variables.
        self.exiting = False
        self.compute_alignment_flag = False

        # Initialize some instance variables.

        self.start()

    def run(self):
        """
        Execute the workflow thread. Its main part is a permanent loop which looks for activity
        flags set by the main gui. When a flag is true, the corresponding action is performed.
        On completion, a signal is emitted.

        :return: -
        """

        # Main workflow loop.
        while not self.exiting:

            # Compute a new image alignment.
            if self.compute_alignment_flag:
                self.compute_alignment_flag = False

                # Detect ORB features and compute descriptors.
                orb = cv2.ORB_create(self.configuration.max_features)
                ny = self.configuration.feature_patch_grid_size_y
                nx = self.configuration.feature_patch_grid_size_x

                keypoints1 = self.getKeypoints(orb, self.gui.image_target_gray, ny, nx)
                keypoints2 = self.getKeypoints(orb, self.gui.image_reference_gray, ny, nx)

                # Compute the descriptors with ORB.
                keypoints1, descriptors1 = orb.compute(self.gui.image_target_gray, keypoints1)
                keypoints2, descriptors2 = orb.compute(self.gui.image_reference_gray, keypoints2)

                # Match features.
                matcher = cv2.BFMatcher(self.configuration.feature_matching_norm,
                                        crossCheck=self.configuration.cross_check)
                matches = matcher.match(descriptors1, descriptors2, None)

                # Sort matches by score
                matches.sort(key=lambda x: x.distance, reverse=False)

                # Remove not so good matches.
                numGoodMatches = int(len(matches) * self.configuration.good_match_fraction)
                matches = matches[:numGoodMatches]

                # Draw top matches
                self.gui.image_matches = cv2.drawMatches(self.gui.image_target, keypoints1,
                                                         self.gui.image_reference, keypoints2,
                                                         matches, None)

                # Extract location of good matches.
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = keypoints1[match.queryIdx].pt
                    points2[i, :] = keypoints2[match.trainIdx].pt

                # Find homography.
                h, mask = cv2.findHomography(points1, points2, self.configuration.match_weighting)

                # Apply homography on target image.
                height, width, channels = self.gui.image_reference.shape
                self.gui.image_rigid_transformed = cv2.warpPerspective(self.gui.image_target,
                                                                       h, (width, height))
                self.gui.image_rigid_transformed_gray = \
                    cv2.cvtColor(self.gui.image_rigid_transformed, cv2.COLOR_BGR2GRAY)

                # Write aligned image to disk.
                # outFilename = "Images/2018-03-24_21-01MEZ_Mond_aligned.jpg"
                # print("Saving aligned image : ", outFilename)
                # cv2.imwrite(outFilename, self.gui.image_rigid_transformed)

                # Signal the GUI that the homography computation is finished.
                self.set_status_signal.emit(3)

                self.gui.image_dewarped = self.deWarp()

                # Write de-warped image to disk.
                # outFilename = "Images/2018-03-24_21-01MEZ_Mond_dewarped.jpg"
                # print("Saving de-warped image : ", outFilename)
                # cv2.imwrite(outFilename, self.gui.image_dewarped)

                # Signal the GUI that the optical flow has been applied to the target image.
                self.set_status_signal.emit(4)

            # Sleep time inserted to limit CPU consumption by idle looping.
            time.sleep(self.gui.configuration.polling_interval)

    def getKeypoints(self, orb, image, ny, nx):
        """
        Detect keypoints in an image. First split the image in a ny x nx grid, and detect the
        keypoints in each patch individually. The individual lists are then concatenated. This way,
        a more uniform keypoint distribution can be achieved.

        :param orb: OpenCV Orb object
        :param image: grayscale image
        :param ny: Number of patches in y direction
        :param nx: Number of patches in x direction
        :return: List of keypoints
        """

        keypoints = []
        for j in range(ny):
            for i in range(nx):
                # Compute an image of the same size as "image" which is black everywhere except
                # for patch (j, i).
                img_patch = self.getPatch(image, ny, nx, j, i)
                # Detect keypoints in single patch.
                kp = orb.detect(img_patch, None)
                # If list is non-epmpty, concatenate it with previous contributions.
                if kp:
                    keypoints += kp
        return keypoints

    def getPatch(self, image, ny, nx, j, i):
        """
        Compute a copy of "image" which is black everywhere except for patch (j, i)
        in a (ny, nx) grid.

        :param image: grayscale image
        :param ny: Number of patches in y direction
        :param nx: Number of patches in x direction
        :param j: Patch coordinate in y direction
        :param i: Patch coordinate in x direction
        :return: Image which is black outside of patch (j, i)
        """

        sh = image.shape
        # Compute coordinate bounds of patch (j, i).
        y_low = int(sh[0] / ny * j)
        y_high = int(sh[0] / ny * (j + 1))
        x_low = int(sh[1] / nx * i)
        x_high = int(sh[1] / nx * (i + 1))
        # Get a black image of the same size as "image".
        new_image = np.zeros(sh, dtype=image.dtype)
        # Copy patch from original image.
        new_image[y_low:y_high, x_low:x_high] = image[y_low:y_high, x_low:x_high]
        return new_image

    def deWarp(self):
        """
        Take the result of the rigid transformation of the target (color) image, compute the
        optical flow between this image and the reference frame, and apply the flow to this image.

        :return: Pixel-wise registered target image.
        """

        # If Gaussian filter is specified, set the flag accordingly. Compute the flow field.
        if self.configuration.use_gaussian_filter:
            flow = cv2.calcOpticalFlowFarneback(self.gui.image_reference_gray,
                                                self.gui.image_rigid_transformed_gray, flow=None,
                                                pyr_scale=self.configuration.pyramid_scale,
                                                levels=self.configuration.levels,
                                                winsize=self.configuration.winsize,
                                                iterations=self.configuration.iterations,
                                                poly_n=self.configuration.poly_n,
                                                poly_sigma=self.configuration.poly_sigma,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        else:
            flow = cv2.calcOpticalFlowFarneback(self.gui.image_reference_gray,
                                                self.gui.image_rigid_transformed_gray, flow=None,
                                                pyr_scale=self.configuration.pyramid_scale,
                                                levels=self.configuration.levels,
                                                winsize=self.configuration.winsize,
                                                iterations=self.configuration.iterations,
                                                poly_n=self.configuration.poly_n,
                                                poly_sigma=self.configuration.poly_sigma)

        # Apply the flow field.
        return self.warp_flow(self.gui.image_rigid_transformed, flow)

    def warp_flow(self, img, flow):
        """
        Apply a flow field to an image.

        :param img: B/W or color image
        :param flow: Flow field resulting from optical flow computation
        :return: Warped image
        """

        # Get image height and width.
        h, w = flow.shape[:2]
        # The flow field so far contains the displacements only. The remap function requires
        # absolute pixel coordinates. Therefore, add the (y, x) pixel coordinates to each point of
        # the flow field.
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        # Apply the remap function with linear interpolation.
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res

