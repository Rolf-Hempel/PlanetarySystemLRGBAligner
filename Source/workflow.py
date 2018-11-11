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
from PyQt5 import QtCore, QtGui


class Workflow(QtCore.QThread):
    """
    The Workflow class creates a thread which runs in parallel to the main gui. Communication
    between the main gui and the workflow thread is realized as follows: Actions in this
    thread are triggered by flags set in the main gui thread. In the reverse direction, this
    thread emits signals which are connected with methods in the main gui.

    """

    # Define the list of signals with which this thread communicates with the main gui.
    set_status_signal = QtCore.pyqtSignal(int)
    set_status_busy_signal = QtCore.pyqtSignal(bool)
    set_error_signal = QtCore.pyqtSignal(str)

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
        self.compute_lrgb_flag = False

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
                self.set_status_busy_signal.emit(True)

                # Check if the rigid transformation step is to be skipped:
                if not self.configuration.skip_rigid_transformation:
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
                    # Load the image into the GUI for display.
                    self.gui.pixmaps[3] = self.create_pixmap(self.gui.image_matches)

                    # Extract location of good matches.
                    points1 = np.zeros((len(matches), 2), dtype=np.float32)
                    points2 = np.zeros((len(matches), 2), dtype=np.float32)

                    for i, match in enumerate(matches):
                        points1[i, :] = keypoints1[match.queryIdx].pt
                        points2[i, :] = keypoints2[match.trainIdx].pt

                    # Find homography.
                    h, mask = cv2.findHomography(points1, points2, self.configuration.match_weighting)

                    # Check the result of homography computation for isotropic scaling and angle
                    # preservation. If deviations are too large, issue a warning.
                    angle_error, scale_error = self.test_homography(h)
                    if angle_error > self.configuration.maximum_allowed_angle_deviation:
                        if scale_error > self.configuration.maximum_allowed_scale_difference:
                            self.set_error_signal.emit("Warning: The rigid transformation shows a "
                                "large discrepancy in (x,y) scaling (%5.1f percent) and deviation "
                                "from orthogonality (%5.1f degrees). It is recommended to try again"
                                " with different rigid transformation parameters. Otherwise, the "
                                "optical flow computation may not give satisfactory results."
                                % (scale_error, angle_error))
                        else:
                            self.set_error_signal.emit("Warning: The rigid transformation shows a "
                                "large deviation from orthogonality (%5.1f degrees). It is "
                                "recommended to try again with different rigid transformation "
                                "parameters. Otherwise, the optical flow computation may not give "
                                "satisfactory results." % (angle_error))
                    elif scale_error > self.configuration.maximum_allowed_scale_difference:
                        self.set_error_signal.emit("Warning: The rigid transformation shows a "
                                "large discrepancy in (x,y) scaling (%5.1f percent). It is "
                                "recommended to try again with different rigid transformation "
                                "parameters. Otherwise, the optical flow computation may not give "
                                "satisfactory results." % (scale_error))

                    # Apply homography on target image.
                    height, width, channels = self.gui.image_reference.shape
                    self.gui.image_rigid_transformed = cv2.warpPerspective(self.gui.image_target,
                                                                           h, (width, height))

                # Before skipping rigid transformation, test if the target image has the correct
                # pixel dimensions. If so, just copy the color input file.
                elif self.gui.image_target.shape[0] == self.gui.image_reference.shape[0] and \
                     self.gui.image_target.shape[1] == self.gui.image_reference.shape[1]:
                    self.gui.image_rigid_transformed = self.gui.image_target
                else:
                    # Input images have different size, issue error message and reset.
                    self.set_error_signal.emit("Error: Rigid transformation cannot be skipped if"
                                               " input images have different size. "
                                               "Uncheck parameter 'skip rigid transformation',"
                                               " or use a different color image file.")
                    # Reset the GUI to the point where only the input files are loaded.
                    self.set_status_busy_signal.emit(False)
                    self.set_status_signal.emit(2)
                    time.sleep(self.gui.configuration.polling_interval)
                    continue

                self.gui.image_rigid_transformed_gray = \
                    cv2.cvtColor(self.gui.image_rigid_transformed, cv2.COLOR_BGR2GRAY)
                self.gui.pixmaps[2] = self.create_pixmap(self.gui.image_rigid_transformed)

                # Signal the GUI that the homography computation is finished.
                self.set_status_signal.emit(3)

                if self.configuration.skip_optical_flow:
                    # Since the rigid transformation always produces an image of the correct size,
                    # this does not have to be checked here again.
                    self.gui.image_dewarped = self.gui.image_rigid_transformed
                else:
                    self.gui.image_dewarped = self.deWarp()
                self.gui.pixmaps[4] = self.create_pixmap(self.gui.image_dewarped)

                # Signal the GUI that the optical flow has been applied to the target image.
                self.set_status_busy_signal.emit(False)
                self.set_status_signal.emit(4)

            # Compute an LRGB composite from the B/W image and the pixelwise aligned color image.
            if self.compute_lrgb_flag:
                self.compute_lrgb_flag = False
                self.set_status_busy_signal.emit(True)

                # Convert de-warped color image to HSV color space.
                image_hsv = cv2.cvtColor(self.gui.image_dewarped, cv2.COLOR_BGR2HSV)

                # Replace V channel with reference grayscale image.
                image_hsv[:, :, 2] = self.gui.image_reference_gray

                # Convert LRGB image back to BGR color space.
                self.gui.image_lrgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

                # Load LRGB image as pixmap into GUI viewer
                self.gui.pixmaps[5] = self.create_pixmap(self.gui.image_lrgb)
                self.set_status_busy_signal.emit(False)
                self.set_status_signal.emit(5)

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

    def test_homography(self, homography_matrix):
        """
        Check the result of homography computation. The difference between the scaling factors in
        x and y direction (in percent) and the deviation from orthogonality (in degrees) are
        computed. If these values exceed certain limits, the homography computation most likely was
        using appropriate keypoints.

        :param homography_matrix: Homography matrix
        :return: (scaling factor deviation, orthogonality violation)
        """

        # Define thre test vectors: One pointing at the origin, and the other two at points on the
        # x and y axes, respectively, at distance 1.
        vec_0 = np.array([0., 0., 1.])
        vec_1x = np.array([0., 1., 1.])
        vec_1y = np.array([1., 0., 1.])

        # Apply the homography transformation on the test vectors, and subtract the first vector
        # from the other two.
        vec_0_hom = homography_matrix.dot(vec_0)
        vec_1x_hom = homography_matrix.dot(vec_1x)
        vec_1y_hom = homography_matrix.dot(vec_1y)
        vec_1x_trans = vec_1x_hom - vec_0_hom
        vec_1y_trans = vec_1y_hom - vec_0_hom

        # Compute the scaling factor in x and y directions, and the relative difference.
        scale_x = np.sqrt(vec_1x_trans[0] ** 2 + vec_1x_trans[1] ** 2)
        scale_y = np.sqrt(vec_1y_trans[0] ** 2 + vec_1y_trans[1] ** 2)
        scale_error = abs((scale_y - scale_x) / scale_x) * 100.

        # Normalize the transformed vectors to length 1, and compute the deviation of the angle
        # between them from orthogonality.
        vec_1x_trans_normalized = vec_1x_trans / scale_x
        vec_1y_trans_normalized = vec_1y_trans / scale_y
        angle_error = np.rad2deg(
            np.arcsin(abs(np.dot(vec_1x_trans_normalized, vec_1y_trans_normalized))))
        return scale_error, angle_error


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

    def create_pixmap(self, cv_image):
        """
        Transform an image in OpenCV color representation (BGR) into a QT pixmap

        :param cv_image: Image array
        :return: QT QPixmap object
        """

        return QtGui.QPixmap(
            QtGui.QImage(cv_image, cv_image.shape[1], cv_image.shape[0], cv_image.shape[1] * 3,
                         QtGui.QImage.Format_RGB888).rgbSwapped())
