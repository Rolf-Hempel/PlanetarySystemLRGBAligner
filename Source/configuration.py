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

import cv2

class Configuration:
    """
    The Configuration class is used to manage all parameters which can be changed or set by the
    user.

    """

    def __init__(self):
        """
        Initialize the configuration object.

        """

        # The version number is displayed on the PSLA main GUI title line.
        self.version = "Planetary System LRGB Aligner 0.5.0"

        # Set internal parameters which cannot be changed by the user.
        self.wait_for_workflow_initialization = 0.1
        self.polling_interval = 0.1
        self.feature_matching_norm = cv2.NORM_HAMMING
        self.cross_check = True

        # Set initial values for parameters which can be modified by the user.
        self.restore_standard_parameters()

        # Set a flag that the configuration has been changed.
        self.configuration_changed = True

    def restore_standard_parameters(self):
        """
        Set configuration parameters to standard values.

        """

        # Parameters to change workflow.
        self.skip_rigid_transformation = False
        self.skip_optical_flow = False

        # Parameters used for rigid transformation.
        # Number of patches in y direction for feature detection. Features are detected in each
        # patch separately. This leads to a more uniform feature distribution.
        self.feature_patch_grid_size_y = 4    # between 1 and 10
        # Number of patches in x direction for feature detection. Features are detected in each
        # patch separately. This leads to a more uniform feature distribution.
        self.feature_patch_grid_size_x = 3    # between 1 and 10
        # Maximal number of features to be detected per patch.
        self.max_features = 100               # between 10 and 200
        # Fraction of detected features to be selected for homography matrix computation.
        self.good_match_fraction = 0.1        # between 0.05 and 1.
        # Weighting method used in solving the over-determined homography matrix problem.
        self.match_weighting = cv2.LMEDS      # either cv2.RANSAC for "random sample consensus"
                                              # or cv2.LMEDS for "least median of squares"

        # Parameters used for optical flow:
        # Image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical
        # pyramid, where each next layer is twice smaller than the previous one.
        self.pyramid_scale = 0.5                # between 0.1 and 0.9
        # Number of pyramid layers including the initial image; levels=1 means that no extra layers
        # are created and only the original images are used.
        self.levels = 1                         # between 1 and 10
        # Averaging window size; larger values increase the algorithm robustness to image noise and
        # give more chances for fast motion detection, but yield more blurred motion field.
        self.winsize = 15                       # between 5 and 40
        # Number of iterations the algorithm does at each pyramid level.
        self.iterations = 1                     # between 1 and 10
        # Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger
        # values mean that the image will be approximated with smoother surfaces, yielding more
        # robust algorithm and more blurred motion field, typically poly_n =5 or 7.
        self.poly_n = 5                         # between 3 and 10
        # Standard deviation of the Gaussian that is used to smooth derivatives used as a basis
        # for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7,
        # a good value would be poly_sigma=1.5.
        self.poly_sigma = 1.1                   # between 1. and 2.
        # Select if the Gaussian winsize * winsize filter should be used instead of a box filter
        # of the same size for optical flow estimation; usually, this option gives a more accurate
        # flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian
        # window should be set to a larger value to achieve the same level of robustness.
        self.use_gaussian_filter = True         # either True or False

