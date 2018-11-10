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

from PyQt5 import QtWidgets, QtCore
import cv2
from parameter_configuration import Ui_ConfigurationDialog

class ConfigurationEditor(QtWidgets.QDialog, Ui_ConfigurationDialog):
    """
    Update the parameters used by MoonPanoramaMaker which are stored in the configuration object.
    The interaction with the user is through the ConfigurationDialog class.

    """

    def __init__(self, configuration, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.configuration = configuration
        self.configuration_changed = False

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Connect local methods with GUI change events.
        self.fpgsy_slider_value.valueChanged['int'].connect(self.fpgsy_changed)
        self.fpgsx_slider_value.valueChanged['int'].connect(self.fpgsx_changed)
        self.mnf_slider_value.valueChanged['int'].connect(self.mnf_changed)
        self.fdf_slider_value.valueChanged['int'].connect(self.fdf_changed)
        self.wm_combobox.addItem("Random Sample Consensus")
        self.wm_combobox.addItem("Least Median of Squares")
        self.wm_combobox.activated[str].connect(self.wm_changed)
        self.ps_slider_value.valueChanged['int'].connect(self.ps_changed)
        self.npl_slider_value.valueChanged['int'].connect(self.npl_changed)
        self.awz_slider_value.valueChanged['int'].connect(self.awz_changed)
        self.ni_slider_value.valueChanged['int'].connect(self.ni_changed)
        self.sn_slider_value.valueChanged['int'].connect(self.sn_changed)
        self.gsd_slider_value.valueChanged['int'].connect(self.gsd_changed)
        self.ugf_checkBox.stateChanged.connect(self.ugf_changed)
        self.srt_checkBox.stateChanged.connect(self.srt_changed)
        self.sof_checkBox.stateChanged.connect(self.sof_changed)
        self.restore_standard_values.clicked.connect(self.restore_standard_parameters)

        self.initialize_widgets_and_local_parameters()

    def initialize_widgets_and_local_parameters(self):
        # Initialize GUI widgets with current configuration parameter values.
        self.fpgsy_slider_value.setValue(self.configuration.feature_patch_grid_size_y)
        self.fpgsx_slider_value.setValue(self.configuration.feature_patch_grid_size_x)
        self.mnf_slider_value.setValue(self.configuration.max_features)
        self.fdf_slider_value.setValue(int(self.configuration.good_match_fraction*100.+0.1))
        if self.configuration.match_weighting == cv2.LMEDS:
            self.wm_combobox.setCurrentIndex(1)
        else:
            self.wm_combobox.setCurrentIndex(0)
        self.ps_slider_value.setValue(int(self.configuration.pyramid_scale*100+0.1))
        self.npl_slider_value.setValue(self.configuration.levels)
        self.awz_slider_value.setValue(self.configuration.winsize)
        self.ni_slider_value.setValue(self.configuration.iterations)
        self.sn_slider_value.setValue(self.configuration.poly_n)
        self.gsd_slider_value.setValue(int(self.configuration.poly_sigma*10+0.1))
        self.ugf_checkBox.setChecked(self.configuration.use_gaussian_filter)
        self.srt_checkBox.setChecked(self.configuration.skip_rigid_transformation)
        self.sof_checkBox.setChecked(self.configuration.skip_optical_flow)

        # Initialize new parameter values with current configuration.
        self.feature_patch_grid_size_y_new = self.configuration.feature_patch_grid_size_y
        self.feature_patch_grid_size_x_new = self.configuration.feature_patch_grid_size_x
        self.max_features_new = self.configuration.max_features
        self.good_match_fraction_new = self.configuration.good_match_fraction
        self.match_weighting_new = self.configuration.match_weighting
        self.pyramid_scale_new = self.configuration.pyramid_scale
        self.levels_new = self.configuration.levels
        self.winsize_new = self.configuration.winsize
        self.iterations_new = self.configuration.iterations
        self.poly_n_new = self.configuration.poly_n
        self.poly_sigma_new = self.configuration.poly_sigma
        self.use_gaussian_filter_new = self.configuration.use_gaussian_filter
        self.skip_rigid_transformation_new = self.configuration.skip_rigid_transformation
        self.skip_optical_flow_new = self.configuration.skip_optical_flow

    def fpgsy_changed(self, value):
        """
        If the GUI changes the parameter "feature_patch_grid_size_y", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.feature_patch_grid_size_y_new = value

    def fpgsx_changed(self, value):
        """
        If the GUI changes the parameter "feature_patch_grid_size_x", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.feature_patch_grid_size_x_new = value

    def mnf_changed(self, value):
        """
        If the GUI changes the parameter "max_features", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.max_features_new = value

    def fdf_changed(self, value):
        """
        If the GUI changes the parameter "good_match_fraction", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.good_match_fraction_new = float(value)/100.

    def wm_changed(self, value):
        """
        If the GUI changes the parameter "match_weighting", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        if value == "Random Sample Consensus":
            self.match_weighting_new = cv2.RANSAC
        elif value == "Least Median of Squares":
            self.match_weighting_new = cv2.LMEDS

    def ps_changed(self, value):
        """
        If the GUI changes the parameter "pyramid_scale", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.pyramid_scale_new = float(value)/100.
        self.ps_label_display.setText(str(self.pyramid_scale_new))

    def npl_changed(self, value):
        """
        If the GUI changes the parameter "levels", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.levels_new = value

    def awz_changed(self, value):
        """
        If the GUI changes the parameter "winsize", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.winsize_new = value

    def ni_changed(self, value):
        """
        If the GUI changes the parameter "iterations", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.iterations_new = value

    def sn_changed(self, value):
        """
        If the GUI changes the parameter "poly_n", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.poly_n_new = value

    def gsd_changed(self, value):
        """
        If the GUI changes the parameter "poly_sigma", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.poly_sigma_new = float(value)/10.
        self.gsd_label_display.setText(str(self.poly_sigma_new))

    def ugf_changed(self, state):
        """
        If the GUI changes the parameter "use_gaussian_filter", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.use_gaussian_filter_new = (state == QtCore.Qt.Checked)

    def srt_changed(self, state):
        """
        If the GUI changes the parameter "skip_rigid_transformation", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.skip_rigid_transformation_new = (state == QtCore.Qt.Checked)

    def sof_changed(self, state):
        """
        If the GUI changes the parameter "skip_optical_flow", assign the new value
        to the "new" variable.

        :param value: Changed input value
        :return: -
        """
        self.skip_optical_flow_new = (state == QtCore.Qt.Checked)

    def restore_standard_parameters(self):
        """
        Reset configuration parameters and GUI widget settings to standard values. Mark
        configuration as changed. This may be too pessimistic, if standard values were not changed
        before.

        :return:
        """
        self.configuration.restore_standard_parameters()
        self.initialize_widgets_and_local_parameters()
        self.configuration_changed = True

    def accept(self):
        """
        If the OK button is clicked and the configuration has been changed, check if values have
        been changed and assign the new values to configuration parameters.


        :return: -
        """

        if self.feature_patch_grid_size_y_new != self.configuration.feature_patch_grid_size_y:
            self.configuration.feature_patch_grid_size_y = self.feature_patch_grid_size_y_new
            self.configuration_changed = True

        if self.feature_patch_grid_size_x_new != self.configuration.feature_patch_grid_size_x:
            self.configuration.feature_patch_grid_size_x = self.feature_patch_grid_size_x_new
            self.configuration_changed = True

        if self.max_features_new != self.configuration.max_features:
            self.configuration.max_features = self.max_features_new
            self.configuration_changed = True

        if self.good_match_fraction_new != self.configuration.good_match_fraction:
            self.configuration.good_match_fraction = self.good_match_fraction_new
            self.configuration_changed = True

        if self.match_weighting_new != self.configuration.match_weighting:
            self.configuration.match_weighting = self.match_weighting_new
            self.configuration_changed = True

        if self.pyramid_scale_new != self.configuration.pyramid_scale:
            self.configuration.pyramid_scale = self.pyramid_scale_new
            self.configuration_changed = True

        if self.levels_new != self.configuration.levels:
            self.configuration.levels = self.levels_new
            self.configuration_changed = True

        if self.winsize_new != self.configuration.winsize:
            self.configuration.winsize = self.winsize_new
            self.configuration_changed = True

        if self.iterations_new != self.configuration.iterations:
            self.configuration.iterations = self.iterations_new
            self.configuration_changed = True

        if self.poly_n_new != self.configuration.poly_n:
            self.configuration.poly_n = self.poly_n_new
            self.configuration_changed = True

        if self.poly_sigma_new != self.configuration.poly_sigma:
            self.configuration.poly_sigma = self.poly_sigma_new
            self.configuration_changed = True

        if self.use_gaussian_filter_new != self.configuration.use_gaussian_filter:
            self.configuration.use_gaussian_filter = self.use_gaussian_filter_new
            self.configuration_changed = True

        if self.skip_rigid_transformation_new != self.configuration.skip_rigid_transformation:
            self.configuration.skip_rigid_transformation = self.skip_rigid_transformation_new
            self.configuration_changed = True

        if self.skip_optical_flow_new != self.configuration.skip_optical_flow:
            self.configuration.skip_optical_flow = self.skip_optical_flow_new
            self.configuration_changed = True

        self.close()

    def reject(self):
        """
        The Cancel button is pressed, discard the changes and close the GUI window.
        :return: -
        """

        self.configuration_changed = False
        self.close()

    def closeEvent(self, event):
        self.close()
