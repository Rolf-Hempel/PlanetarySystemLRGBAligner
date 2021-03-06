# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'parameter_configuration.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ConfigurationDialog(object):
    def setupUi(self, ConfigurationDialog):
        ConfigurationDialog.setObjectName("ConfigurationDialog")
        ConfigurationDialog.resize(921, 543)
        font = QtGui.QFont()
        font.setPointSize(10)
        ConfigurationDialog.setFont(font)
        self.GridLayout = QtWidgets.QGridLayout(ConfigurationDialog)
        self.GridLayout.setVerticalSpacing(22)
        self.GridLayout.setObjectName("GridLayout")
        self.ugf_checkBox = QtWidgets.QCheckBox(ConfigurationDialog)
        self.ugf_checkBox.setChecked(True)
        self.ugf_checkBox.setObjectName("ugf_checkBox")
        self.GridLayout.addWidget(self.ugf_checkBox, 5, 7, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.GridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.awz_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.awz_label_parameter.setObjectName("awz_label_parameter")
        self.GridLayout.addWidget(self.awz_label_parameter, 3, 7, 1, 1)
        self.fpgsx_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.fpgsx_label_display.setObjectName("fpgsx_label_display")
        self.GridLayout.addWidget(self.fpgsx_label_display, 2, 4, 1, 1)
        self.ni_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.ni_label_parameter.setObjectName("ni_label_parameter")
        self.GridLayout.addWidget(self.ni_label_parameter, 4, 7, 1, 1)
        self.sn_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.sn_label_display.setObjectName("sn_label_display")
        self.GridLayout.addWidget(self.sn_label_display, 6, 11, 1, 1)
        self.fdf_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.fdf_label_display.setObjectName("fdf_label_display")
        self.GridLayout.addWidget(self.fdf_label_display, 4, 4, 1, 1)
        self.ni_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.ni_label_display.setObjectName("ni_label_display")
        self.GridLayout.addWidget(self.ni_label_display, 4, 11, 1, 1)
        self.fpgsy_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.fpgsy_slider_value.setMinimum(1)
        self.fpgsy_slider_value.setMaximum(10)
        self.fpgsy_slider_value.setPageStep(2)
        self.fpgsy_slider_value.setProperty("value", 1)
        self.fpgsy_slider_value.setSliderPosition(1)
        self.fpgsy_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.fpgsy_slider_value.setObjectName("fpgsy_slider_value")
        self.GridLayout.addWidget(self.fpgsy_slider_value, 1, 2, 1, 1)
        self.ps_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.ps_label_display.setObjectName("ps_label_display")
        self.GridLayout.addWidget(self.ps_label_display, 1, 11, 1, 1)
        self.gsd_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.gsd_label_parameter.setObjectName("gsd_label_parameter")
        self.GridLayout.addWidget(self.gsd_label_parameter, 7, 7, 1, 1)
        self.label_4 = QtWidgets.QLabel(ConfigurationDialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.GridLayout.addWidget(self.label_4, 8, 0, 1, 1)
        self.fpgsx_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.fpgsx_label_parameter.setObjectName("fpgsx_label_parameter")
        self.GridLayout.addWidget(self.fpgsx_label_parameter, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(ConfigurationDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.GridLayout.addWidget(self.label_2, 0, 7, 1, 5)
        self.fpgsx_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.fpgsx_slider_value.setMinimum(1)
        self.fpgsx_slider_value.setMaximum(10)
        self.fpgsx_slider_value.setPageStep(2)
        self.fpgsx_slider_value.setProperty("value", 1)
        self.fpgsx_slider_value.setSliderPosition(1)
        self.fpgsx_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.fpgsx_slider_value.setObjectName("fpgsx_slider_value")
        self.GridLayout.addWidget(self.fpgsx_slider_value, 2, 2, 1, 1)
        self.fpgsy_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.fpgsy_label_parameter.setObjectName("fpgsy_label_parameter")
        self.GridLayout.addWidget(self.fpgsy_label_parameter, 1, 0, 1, 1)
        self.npl_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.npl_label_display.setObjectName("npl_label_display")
        self.GridLayout.addWidget(self.npl_label_display, 2, 11, 1, 1)
        self.line = QtWidgets.QFrame(ConfigurationDialog)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.GridLayout.addWidget(self.line, 0, 5, 12, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.GridLayout.addItem(spacerItem1, 1, 6, 1, 1)
        self.ps_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.ps_label_parameter.setObjectName("ps_label_parameter")
        self.GridLayout.addWidget(self.ps_label_parameter, 1, 7, 1, 1)
        self.mnf_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.mnf_slider_value.setMinimum(1)
        self.mnf_slider_value.setMaximum(200)
        self.mnf_slider_value.setProperty("value", 97)
        self.mnf_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.mnf_slider_value.setObjectName("mnf_slider_value")
        self.GridLayout.addWidget(self.mnf_slider_value, 3, 2, 1, 1)
        self.fdf_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.fdf_slider_value.setMinimum(5)
        self.fdf_slider_value.setProperty("value", 9)
        self.fdf_slider_value.setSliderPosition(9)
        self.fdf_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.fdf_slider_value.setObjectName("fdf_slider_value")
        self.GridLayout.addWidget(self.fdf_slider_value, 4, 2, 1, 1)
        self.fpgsy_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.fpgsy_label_display.setObjectName("fpgsy_label_display")
        self.GridLayout.addWidget(self.fpgsy_label_display, 1, 4, 1, 1)
        self.npl_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.npl_slider_value.setMinimum(1)
        self.npl_slider_value.setMaximum(10)
        self.npl_slider_value.setPageStep(2)
        self.npl_slider_value.setProperty("value", 2)
        self.npl_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.npl_slider_value.setObjectName("npl_slider_value")
        self.GridLayout.addWidget(self.npl_slider_value, 2, 9, 1, 1)
        self.npl_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.npl_label_parameter.setObjectName("npl_label_parameter")
        self.GridLayout.addWidget(self.npl_label_parameter, 2, 7, 1, 1)
        self.fdf_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.fdf_label_parameter.setObjectName("fdf_label_parameter")
        self.GridLayout.addWidget(self.fdf_label_parameter, 4, 0, 1, 1)
        self.sn_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.sn_slider_value.setMinimum(3)
        self.sn_slider_value.setMaximum(10)
        self.sn_slider_value.setPageStep(2)
        self.sn_slider_value.setProperty("value", 6)
        self.sn_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.sn_slider_value.setObjectName("sn_slider_value")
        self.GridLayout.addWidget(self.sn_slider_value, 6, 9, 1, 1)
        self.sn_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.sn_label_parameter.setObjectName("sn_label_parameter")
        self.GridLayout.addWidget(self.sn_label_parameter, 6, 7, 1, 1)
        self.gsd_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.gsd_slider_value.setMinimum(10)
        self.gsd_slider_value.setMaximum(20)
        self.gsd_slider_value.setPageStep(2)
        self.gsd_slider_value.setProperty("value", 10)
        self.gsd_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.gsd_slider_value.setObjectName("gsd_slider_value")
        self.GridLayout.addWidget(self.gsd_slider_value, 7, 9, 1, 1)
        self.ps_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.ps_slider_value.setMinimum(10)
        self.ps_slider_value.setMaximum(90)
        self.ps_slider_value.setProperty("value", 49)
        self.ps_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.ps_slider_value.setObjectName("ps_slider_value")
        self.GridLayout.addWidget(self.ps_slider_value, 1, 9, 1, 1)
        self.gsd_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.gsd_label_display.setObjectName("gsd_label_display")
        self.GridLayout.addWidget(self.gsd_label_display, 7, 11, 1, 1)
        self.awz_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.awz_slider_value.setMinimum(5)
        self.awz_slider_value.setMaximum(40)
        self.awz_slider_value.setPageStep(5)
        self.awz_slider_value.setProperty("value", 14)
        self.awz_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.awz_slider_value.setObjectName("awz_slider_value")
        self.GridLayout.addWidget(self.awz_slider_value, 3, 9, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.GridLayout.addItem(spacerItem2, 1, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.GridLayout.addItem(spacerItem3, 1, 8, 1, 1)
        self.label = QtWidgets.QLabel(ConfigurationDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.GridLayout.addWidget(self.label, 0, 0, 1, 5)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.GridLayout.addItem(spacerItem4, 1, 10, 1, 1)
        self.mnf_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.mnf_label_parameter.setObjectName("mnf_label_parameter")
        self.GridLayout.addWidget(self.mnf_label_parameter, 3, 0, 1, 1)
        self.awz_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.awz_label_display.setObjectName("awz_label_display")
        self.GridLayout.addWidget(self.awz_label_display, 3, 11, 1, 1)
        self.mnf_label_display = QtWidgets.QLabel(ConfigurationDialog)
        self.mnf_label_display.setObjectName("mnf_label_display")
        self.GridLayout.addWidget(self.mnf_label_display, 3, 4, 1, 1)
        self.srt_checkBox = QtWidgets.QCheckBox(ConfigurationDialog)
        self.srt_checkBox.setObjectName("srt_checkBox")
        self.GridLayout.addWidget(self.srt_checkBox, 9, 0, 1, 1)
        self.sof_checkBox = QtWidgets.QCheckBox(ConfigurationDialog)
        self.sof_checkBox.setObjectName("sof_checkBox")
        self.GridLayout.addWidget(self.sof_checkBox, 10, 0, 1, 1)
        self.wm_label_parameter = QtWidgets.QLabel(ConfigurationDialog)
        self.wm_label_parameter.setObjectName("wm_label_parameter")
        self.GridLayout.addWidget(self.wm_label_parameter, 6, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(ConfigurationDialog)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.GridLayout.addWidget(self.line_2, 7, 0, 1, 5)
        self.wm_combobox = QtWidgets.QComboBox(ConfigurationDialog)
        self.wm_combobox.setCurrentText("")
        self.wm_combobox.setMaxVisibleItems(2)
        self.wm_combobox.setMaxCount(2)
        self.wm_combobox.setMinimumContentsLength(0)
        self.wm_combobox.setObjectName("wm_combobox")
        self.GridLayout.addWidget(self.wm_combobox, 6, 2, 1, 3)
        self.line_3 = QtWidgets.QFrame(ConfigurationDialog)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.GridLayout.addWidget(self.line_3, 8, 6, 1, 6)
        self.ni_slider_value = QtWidgets.QSlider(ConfigurationDialog)
        self.ni_slider_value.setMinimum(1)
        self.ni_slider_value.setMaximum(10)
        self.ni_slider_value.setPageStep(2)
        self.ni_slider_value.setProperty("value", 2)
        self.ni_slider_value.setOrientation(QtCore.Qt.Horizontal)
        self.ni_slider_value.setObjectName("ni_slider_value")
        self.GridLayout.addWidget(self.ni_slider_value, 4, 9, 1, 1)
        self.restore_standard_values = QtWidgets.QPushButton(ConfigurationDialog)
        self.restore_standard_values.setObjectName("restore_standard_values")
        self.GridLayout.addWidget(self.restore_standard_values, 9, 9, 1, 3)
        self.buttonBox = QtWidgets.QDialogButtonBox(ConfigurationDialog)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.GridLayout.addWidget(self.buttonBox, 10, 9, 1, 3)
        self.GridLayout.setColumnStretch(0, 5)
        self.GridLayout.setColumnStretch(1, 1)
        self.GridLayout.setColumnStretch(2, 10)
        self.GridLayout.setColumnStretch(3, 1)
        self.GridLayout.setColumnStretch(4, 3)
        self.GridLayout.setColumnStretch(5, 1)
        self.GridLayout.setColumnStretch(6, 1)
        self.GridLayout.setColumnStretch(7, 5)
        self.GridLayout.setColumnStretch(8, 1)
        self.GridLayout.setColumnStretch(9, 10)
        self.GridLayout.setColumnStretch(10, 1)
        self.GridLayout.setColumnStretch(11, 3)

        self.retranslateUi(ConfigurationDialog)
        self.wm_combobox.setCurrentIndex(-1)
        self.fpgsy_slider_value.valueChanged['int'].connect(self.fpgsy_label_display.setNum)
        self.fpgsx_slider_value.valueChanged['int'].connect(self.fpgsx_label_display.setNum)
        self.mnf_slider_value.valueChanged['int'].connect(self.mnf_label_display.setNum)
        self.ps_slider_value.valueChanged['int'].connect(self.ps_label_display.setNum)
        self.fdf_slider_value.valueChanged['int'].connect(self.fdf_label_display.setNum)
        self.npl_slider_value.valueChanged['int'].connect(self.npl_label_display.setNum)
        self.awz_slider_value.valueChanged['int'].connect(self.awz_label_display.setNum)
        self.ni_slider_value.valueChanged['int'].connect(self.ni_label_display.setNum)
        self.sn_slider_value.valueChanged['int'].connect(self.sn_label_display.setNum)
        self.gsd_slider_value.valueChanged['int'].connect(self.gsd_label_display.setNum)
        QtCore.QMetaObject.connectSlotsByName(ConfigurationDialog)

    def retranslateUi(self, ConfigurationDialog):
        _translate = QtCore.QCoreApplication.translate
        ConfigurationDialog.setWindowTitle(_translate("ConfigurationDialog", "Parameter Configuration"))
        self.ugf_checkBox.setToolTip(_translate("ConfigurationDialog", "Select if the Gaussian filter should be used instead of a box filter of the same size for optical flow estimation;\n"
"the parameter below gives the filter size in both directions; usually, this option gives a more accurate flow\n"
"than with a box filter, at the cost of lower speed; normally, the filter size for a Gaussian window should be set\n"
"to a larger value to achieve the same level of robustness."))
        self.ugf_checkBox.setText(_translate("ConfigurationDialog", "Use Gaussian filter"))
        self.awz_label_parameter.setToolTip(_translate("ConfigurationDialog", "Larger values increase the algorithm robustness to image noise and give\n"
"more chances for fast motion detection, but yield more blurred motion field."))
        self.awz_label_parameter.setText(_translate("ConfigurationDialog", "Averaging window size"))
        self.fpgsx_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.ni_label_parameter.setToolTip(_translate("ConfigurationDialog", "Number of iterations the algorithm does at each pyramid level."))
        self.ni_label_parameter.setText(_translate("ConfigurationDialog", "Number of iterations"))
        self.sn_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.fdf_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.ni_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.ps_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.gsd_label_parameter.setToolTip(_translate("ConfigurationDialog", "Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for\n"
"the polynomial expansion; for a neighborhood size of 5, you can choose 1.1; for a size\n"
"of 7, a good value would be 1.5."))
        self.gsd_label_parameter.setText(_translate("ConfigurationDialog", "Gaussian standard deviation\n"
"for derivative smoothing"))
        self.label_4.setToolTip(_translate("ConfigurationDialog", "With these parameters parts of the computing workflow can be skipped."))
        self.label_4.setText(_translate("ConfigurationDialog", "Workflow Parameters"))
        self.fpgsx_label_parameter.setToolTip(_translate("ConfigurationDialog", "Number of patches in x direction for feature detection.\n"
"Features are detected in each patch separately.\n"
"This leads to a more uniform feature distribution."))
        self.fpgsx_label_parameter.setText(_translate("ConfigurationDialog", "Number of patches in x direction\n"
"for feature detection"))
        self.label_2.setToolTip(_translate("ConfigurationDialog", "Parameters used in the second (fine-tuning) phase. In each pixel the color image\n"
"is warped such that it exactly matches the B/W image."))
        self.label_2.setText(_translate("ConfigurationDialog", "Optical Flow Parameters"))
        self.fpgsy_label_parameter.setToolTip(_translate("ConfigurationDialog", "Feature detection on the entire image often leads to large void areas. As a remedy,\n"
" the image can be split in a grid of patches, and feature detection is done for each patch separately.\n"
"This leads to a more uniform feature distribution. This parameter specifies the number of patches in y direction."))
        self.fpgsy_label_parameter.setText(_translate("ConfigurationDialog", "Number of patches in y direction\n"
"for feature detection"))
        self.npl_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.ps_label_parameter.setToolTip(_translate("ConfigurationDialog", "A value of 0.5 means a classical pyramid, where each\n"
"next layer is half as large as the previous one."))
        self.ps_label_parameter.setText(_translate("ConfigurationDialog", "Image scale to build\n"
"pyramids for each image"))
        self.fpgsy_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.npl_label_parameter.setToolTip(_translate("ConfigurationDialog", "A value of 1 means that no extra layers are created and only the original images are used."))
        self.npl_label_parameter.setText(_translate("ConfigurationDialog", "Number of pyramid layers"))
        self.fdf_label_parameter.setToolTip(_translate("ConfigurationDialog", "Including doubtful matches (large fraction value) can lead to a low transformation quality.\n"
"If the value is too small, it may not be possible to compute the transformation at all."))
        self.fdf_label_parameter.setText(_translate("ConfigurationDialog", "Fraction of detected features to\n"
"be selected for homography\n"
"matrix computation (%)"))
        self.sn_label_parameter.setToolTip(_translate("ConfigurationDialog", "Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger\n"
"values mean that the image will be approximated with smoother surfaces, yielding more\n"
"robust algorithm and a more blurred motion field, typically values are 5 or 7."))
        self.sn_label_parameter.setText(_translate("ConfigurationDialog", "Size of neighborhood"))
        self.gsd_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.label.setToolTip(_translate("ConfigurationDialog", "Parameters used in the first matching phase. In this phase, the color image\n"
"is rotated, shifted and stretched to match the B/W image."))
        self.label.setText(_translate("ConfigurationDialog", "Rigid Transformation Parameters"))
        self.mnf_label_parameter.setToolTip(_translate("ConfigurationDialog", "Maximal number of features to be detected per patch"))
        self.mnf_label_parameter.setText(_translate("ConfigurationDialog", "Maximal number of features\n"
"to be detected per patch"))
        self.awz_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.mnf_label_display.setText(_translate("ConfigurationDialog", "TextLabel"))
        self.srt_checkBox.setToolTip(_translate("ConfigurationDialog", "If the input color image matches the B/W image very accurately already (and has the same shape), the rigid transformation step can be skipped.\n"
"In this case only the optical flow is computed for fine-tuning."))
        self.srt_checkBox.setText(_translate("ConfigurationDialog", "Skip rigid transformation"))
        self.sof_checkBox.setToolTip(_translate("ConfigurationDialog", "Check this option if only rigid transformation should be applied to the color image.\n"
"Usually this leads to a less accurate match."))
        self.sof_checkBox.setText(_translate("ConfigurationDialog", "Skip optical Flow"))
        self.wm_label_parameter.setToolTip(_translate("ConfigurationDialog", "Weighting method used in solving the over-determined homography matrix problem"))
        self.wm_label_parameter.setText(_translate("ConfigurationDialog", "Weighting method"))
        self.restore_standard_values.setToolTip(_translate("ConfigurationDialog", "Reset parameters to original values. In most cases they should give satisfactory results."))
        self.restore_standard_values.setText(_translate("ConfigurationDialog", "Restore standard values"))

