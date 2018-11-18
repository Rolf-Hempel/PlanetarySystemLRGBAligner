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

import sys
from pathlib import Path
from time import sleep

import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets

from configuration import Configuration
from configuration_editor import ConfigurationEditor
from main_gui import Ui_MainWindow
from photo_viewer import PhotoViewer
from workflow import Workflow


class LrgbAligner(QtWidgets.QMainWindow):
    """
    This class is the main class of the "Planetary System LRGB Aligner" software. It implements
    the main GUI for the communication with the user. It creates the workflow thread which controls
    all program activities asynchronously.

    """

    def __init__(self, parent=None):
        """
        Initialize the Planetary System LRGB Aligner environment.

        :param parent: None
        """

        # The (generated) QtGui class is contained in module main_gui.py.
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Insert the photo viewer into the main GUI.
        self.ImageWindow = PhotoViewer(self)
        self.ImageWindow.setObjectName("ImageWindow")
        self.ui.verticalLayout_3.insertWidget(1, self.ImageWindow, stretch=1)

        # Connect main GUI events with method invocations.
        self.ui.buttonLoadBW.clicked.connect(self.load_bw_image)
        self.ui.buttonLoadColor.clicked.connect(self.load_color_image)
        self.ui.buttonRegistration.clicked.connect(self.compute_registration)
        self.ui.buttonComputeLRGB.clicked.connect(self.compute_lrgb)
        self.ui.buttonSetConfigParams.clicked.connect(self.edit_configuration)
        self.ui.buttonSaveRegisteredColorImage.clicked.connect(self.save_registered_image)
        self.ui.buttonSaveLRGB.clicked.connect(self.save_lrgb_image)
        self.ui.buttonExit.clicked.connect(self.closeEvent)

        self.ui.radioShowBW.clicked.connect(lambda: self.show_pixmap(pixmap_index=0))
        self.ui.radioShowColorOrig.clicked.connect(lambda: self.show_pixmap(pixmap_index=1))
        self.ui.radioShowColorRigidTransform.clicked.connect(
            lambda: self.show_pixmap(pixmap_index=2))
        self.ui.radioShowMatches.clicked.connect(lambda: self.show_pixmap(pixmap_index=3))
        self.ui.radioShowColorOptFlow.clicked.connect(lambda: self.show_pixmap(pixmap_index=4))
        self.ui.radioShowLRGB.clicked.connect(lambda: self.show_pixmap(pixmap_index=5))

        # Initialize the path to the home directory.
        self.current_dir = str(Path.home())

        # Initialize instance variables.
        self.image_reference = None
        self.image_reference_8bit_gray = None
        self.image_target = None
        self.image_target_8bit_gray = None
        self.image_dewarped = None
        self.image_lrgb = None
        self.pixmaps = [None, None, None, None, None, None]
        self.current_pixmap_index = None

        # Initialize status variables
        self.status_list = [False, False, False, False, False, False, False, False]
        self.status_pointer = {"initialized": 0,
                               "bw_loaded": 1,
                               "color_loaded": 2,
                               "rigid_transformed": 3,
                               "optical_flow_computed": 4,
                               "lrgb_computed": 5,
                               "results_saved": 6}

        self.radio_buttons = [self.ui.radioShowBW,                   # 0
                              self.ui.radioShowColorOrig,            # 1
                              self.ui.radioShowColorRigidTransform,  # 2
                              self.ui.radioShowMatches,              # 3
                              self.ui.radioShowColorOptFlow,         # 4
                              self.ui.radioShowLRGB]                 # 5

        self.control_buttons = [self.ui.buttonSetConfigParams,            # 0
                                self.ui.buttonLoadBW,                     # 1
                                self.ui.buttonLoadColor,                  # 2
                                self.ui.buttonRegistration,               # 3
                                self.ui.buttonSaveRegisteredColorImage,   # 4
                                self.ui.buttonComputeLRGB,                # 5
                                self.ui.buttonSaveLRGB,                   # 6
                                self.ui.buttonExit]                       # 7

        self.max_button = [0, 1, 2, 4, 5, 6, 6]
        self.max_control_button = [2, 3, 4, 4, 6, 7, 8]

        self.status_busy = False

        # Create configuration object and set configuration parameters to standard values.
        self.configuration = Configuration()

        # Write the program version into the window title.
        self.setWindowTitle(self.configuration.version)

        # Start the workflow thread. It controls the computations and control of external devices.
        # By decoupling those activities from the main thread, the GUI is kept from freezing during
        # long-running activities.
        self.workflow = Workflow(self)
        sleep(self.configuration.wait_for_workflow_initialization)

        # The workflow thread sends signals during computations. Connect those signals with the
        # appropriate GUI activity.
        self.workflow.set_status_busy_signal.connect(self.set_busy)
        self.workflow.set_status_signal.connect(self.set_status)
        self.workflow.set_error_signal.connect(self.show_error_message)

        # Reset downstream status flags.
        self.set_status(0)

    def edit_configuration(self):
        """
        This method is invoked with the "Set configuration parameters" GUI button. Open the
        configuration editor. If the configuration is changed, set the flag which tells the
        workflow thread to repeat the alignment process.

        :return: -
        """

        editor = ConfigurationEditor(self.configuration)
        editor.exec_()
        if editor.configuration_changed:
            # If parameters have changed, a new alignment has to be computed. If both images are
            # available, set process status to 2.
            if self.current_status > 2:
                self.set_status(2)

    def load_bw_image(self):
        """
        Load the B/W reference image from a file. Keep it together with a grayscale version.

        :return: -
        """

        try:
            self.image_reference, self.image_reference_8bit_gray, self.image_reference_8bit_color = \
                self.load_image("Load B/W reference image", 0, color=False)
            self.ui.radioShowBW.setChecked(True)
            self.current_pixmap_index = 0
            self.set_status(1)
        except:
            pass

    def load_color_image(self):
        """
        Load the color target image from a file. Keep it together with a grayscale version.

        :return: -
        """

        try:
            self.image_target, self.image_target_8bit_gray, self.image_target_8bit_color = \
                self.load_image("Load color image to be registered", 1, color=True)
            self.ui.radioShowColorOrig.setChecked(True)
            self.current_pixmap_index = 1
            self.set_status(2)
        except:
            pass

    def load_image(self, message, pixmap_index, color=False):
        """
        Read an image from a file. Convert it to color mode if optional parameter is set to True.

        :param pixmap_index: Index into the list of pixel maps used to show images in GUI.
        :param color: If True, convert image to color mode. If False, convert it to grayscale.
        :return: 3-tupel with numpy arrays with image data in three versions:
            - Original depth and color / grayscale
            - 8bit grayscale
            - 8bit color
        """

        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getOpenFileName(self, message, self.current_dir,
                                                         "Images (*.tif *.tiff *.png *.jpg)",
                                                         options=options)
        file_name = filename[0]
        if file_name == '':
            raise Exception("File dialog aborted")
        # Remember the current directory for next file dialog.
        self.current_dir = str(Path(file_name).parents[0])
        if color:
            image_read = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            if image_read.dtype == np.uint16:
                image_read_8bit_color = (image_read / 256).astype('uint8')
            else:
                image_read_8bit_color = image_read
            image_read_8bit_gray = cv2.cvtColor(image_read_8bit_color, cv2.COLOR_BGR2GRAY)
        else:
            image_read = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            # If color image, convert to grayscale.
            if len(image_read.shape) == 3:
                image_read = cv2.cv2.cvtColor(image_read, cv2.COLOR_BayerRG2GRAY)
            if image_read.dtype == np.uint16:
                image_read_8bit_gray = cv2.convertScaleAbs(image_read, alpha=(255.0 / 65535.0))
            else:
                image_read_8bit_gray = image_read
            image_read_8bit_color = cv2.cvtColor(image_read_8bit_gray, cv2.COLOR_GRAY2BGR)

        # Convert image into QT pixmel map, store it in list and load it into GUI viewer.
        self.pixmaps[pixmap_index] = self.create_pixmap(image_read_8bit_color)
        self.ImageWindow.setPhoto(self.pixmaps[pixmap_index])
        self.ImageWindow.fitInView()
        return image_read, image_read_8bit_gray, image_read_8bit_color

    def create_pixmap(self, cv_image):
        """
        Transform an image in OpenCV color representation (BGR) into a QT pixmap

        :param cv_image: Image array
        :return: QT QPixmap object
        """

        return QtGui.QPixmap(
            QtGui.QImage(cv_image, cv_image.shape[1], cv_image.shape[0], cv_image.shape[1] * 3,
                         QtGui.QImage.Format_RGB888).rgbSwapped())

    def show_pixmap(self, pixmap_index=None):
        """
        Load a pixmap into the GUI image viewer. Adapt the view scale according to the relative
        sizes of the new and old pixmaps.

        :param pixmap_index: Index of the selected pixmap in the list. If not selected, the
                             current index is taken.
        :return: -
        """

        if pixmap_index is None:
            pixmap_index = self.current_pixmap_index

        if self.pixmaps[pixmap_index] is not None:
            self.current_pixmap_index = pixmap_index

            # Get the ratio of old pixmap and viewport sizes.
            factor_old = self.ImageWindow.fitInView(scale=False)

            # Load the new pixmap.
            self.ImageWindow.setPhoto(self.pixmaps[pixmap_index])

            # Get the ratio of new pixmap and viewport sizes.
            factor_new = self.ImageWindow.fitInView(scale=False)

            if factor_old is not None:
                # Scale the view by the relative size factors.
                factor = factor_new / factor_old
                self.ImageWindow.scale(factor, factor)

    def compute_registration(self):
        """
        If both B/W and color images are available, start the registration process.

        :return: -
        """

        if self.image_reference is not None and self.image_target is not None:
            # Tell the workflow thread to compute a new alignment.
            self.workflow.compute_alignment_flag = True

    def compute_lrgb(self):
        """
        If both B/W and color images are available, start the registration process.

        :return: -
        """

        if self.image_reference is not None and self.image_dewarped is not None:
            # Tell the workflow thread to compute a new alignment.
            self.workflow.compute_lrgb_flag = True

    def save_registered_image(self):
        """
        Open a file chooser dialog and save the de-warped image.

        :return: -
        """

        try:
            self.save_image("Select location to store the registered color image",
                            self.image_dewarped)
            # Udpate the status line.
            self.set_status(6)
        except:
            pass

    def save_lrgb_image(self):
        """
        Open a file chooser dialog and save the combined LRBG image.

        :return: -
        """

        try:
            self.save_image("Select location to store the combined LRGB image",
                            self.image_lrgb)
            # Udpate the status line.
            self.set_status(6)
        except:
            pass

    def save_image(self, message, image):
        """
        Open a file chooser. If a valid file name is selected, store the image to that location.

        :param message: Text to be displayed in the file chooser window.
        :param image: Image file
        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getSaveFileName(self, message, self.current_dir,
                                                         "Images (*.tif *.tiff *.png *.jpg)",
                                                         options=options)
        # Store image only if the chooser did not return with a cancel.
        file_name = filename[0]
        if file_name != "":
            my_file = Path(file_name)
            # Remember the current directory for next file dialog.
            self.current_dir = str(my_file.parents[0])
            if my_file.is_file():
                os.remove(str(my_file))
            if my_file.suffix == '.tif' or my_file.suffix == '.tiff':
                cv2.imwrite(file_name, image)
            elif image.dtype == np.uint16:
                image_8bit = (image / 256).astype('uint8')
                cv2.imwrite(file_name, image_8bit)
            else:
                cv2.imwrite(file_name, image)

        else:
            raise Exception("File dialog aborted")

    def set_busy(self, busy):
        """
        Set the "busy" status flag and update the status bar.

        :param busy: True, if the workflow thread is active in a computation. False, otherwise.
        :return: -
        """

        if busy:
            for button in self.control_buttons[0:7]:
                button.setEnabled(False)
        else:
            self.set_status(self.current_status)
        self.status_busy = busy
        self.set_statusbar()

    def set_status(self, status):
        """
        Enable radio buttons to show images in GUI and set the status bar according to the
        workflow status.

        :param status: Status variable
        :return: -
        """

        # Store current status.
        self.current_status = status

        # Set the current status.
        self.status_list[status] = True

        # Reset all downstream status variables to False.
        self.status_list[status + 1:] = [False] * (len(self.status_list) - status - 1)

        if status != 6:
            # Enable radio buttons which can be used at this point:
            for button in self.radio_buttons[:self.max_button[status]]:
                button.setEnabled(True)
            # Disable the radio buttons for showing images which do not exist at this point.
            for button in self.radio_buttons[self.max_button[status]:]:
                button.setEnabled(False)

            if not self.status_busy:
                # Enable control buttons which can be used at this point:
                for button in self.control_buttons[:self.max_control_button[status]]:
                    button.setEnabled(True)
                # Disable the control buttons which do not make sense at this point.
                for button in self.control_buttons[self.max_control_button[status]:-1]:
                    button.setEnabled(False)

        if self.configuration.skip_rigid_transformation:
            self.ui.radioShowColorRigidTransform.setEnabled(False)
            self.ui.radioShowMatches.setEnabled(False)
        if self.configuration.skip_optical_flow:
            self.ui.radioShowColorOptFlow.setEnabled(False)

        # Refresh the image viewer.
        if self.current_pixmap_index is not None:
            self.show_pixmap()

        # Update the status bar.
        self.set_statusbar()

    def set_statusbar(self):
        """
        The status bar at the bottom of the main GUI summarizes various infos on the process status.
        Read out flags to decide which infos to present. The status information is concatenated into
        a single "status_text" which eventually is written into the main GUI status bar.

        :return: -
        """

        status_text = ""

        # Tell if input images are loaded.
        if self.status_list[self.status_pointer["initialized"]]:
            status_text += "Process initialized"

        # Tell if input images are loaded.
        if self.status_list[self.status_pointer["bw_loaded"]]:
            if self.status_list[self.status_pointer["color_loaded"]]:
                status_text += ", B/W reference and color frames loaded"
            else:
                status_text += ", B/W reference frame loaded"

        # Tell if rigid transformation is done.
        if not self.configuration.skip_rigid_transformation:
            if self.status_list[self.status_pointer["rigid_transformed"]]:
                status_text += ", rigid transformation computed"
        # Tell if optical flow has been computed.
        if not self.configuration.skip_optical_flow:
            if self.status_list[self.status_pointer["optical_flow_computed"]]:
                status_text += ", images pixel-wise aligned"

        # Tell if the LRGB image is computed.
        if self.status_list[self.status_pointer["lrgb_computed"]]:
            status_text += ", LRGB image computed"

        # Tell if results are written to disk.
        if self.status_list[self.status_pointer["results_saved"]]:
            status_text += ", results written to disk"

        # Tell if the workflow thread is busy at this point.
        if self.status_busy:
            status_text += ", busy"

        # Write the complete message to the status bar.
        self.ui.statusbar.showMessage(status_text)

    def show_error_message(self, message):
        """
        Show an error message. This method is invoked from the workflow thread via the
        "set_error_signal" signal.

        :param message: Error message to be displayed
        :return: -
        """

        error_dialog = QtWidgets.QErrorMessage(self)
        error_dialog.setMinimumSize(400, 0)
        error_dialog.setWindowTitle(self.configuration.version)
        error_dialog.showMessage(message)

    def closeEvent(self, evnt):
        """
        This event is triggered when the user closes the main window by clicking on the cross in
        the window corner.

        :param evnt: event object
        :return: -
        """
        self.workflow.exiting = True
        sleep(4. * self.configuration.polling_interval)
        sys.exit(0)


if __name__ == "__main__":
    # The following four lines are a workaround to make PyInstaller work. Remove them when the
    # PyInstaller issue is fixed. Additionally, the following steps are required to get the
    # program running on Linux:
    #
    # - Add "export QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb" to file .bashrc.
    #
    # - There is still a problem with fonts: PyInstaller seems to hardcode the path to fonts
    #   which do not make sense on another computer. This leads to error messages
    #   "Fontconfig error: Cannot load default config file", and a standard font is used
    #   instead.
    #
    # To run the PyInstaller, open a Terminal in PyCharm and enter
    # "pyinstaller moon_panorama_maker_windows.spec" on Windows, or
    # "pyinstaller moon_panorama_maker_linux.spec" on Linux
    #
    import os

    if getattr(sys, 'frozen', False):
        here = os.path.dirname(sys.executable)
        sys.path.insert(1, here)

    app = QtWidgets.QApplication(sys.argv)
    myapp = LrgbAligner()
    myapp.show()
    sys.exit(app.exec_())
