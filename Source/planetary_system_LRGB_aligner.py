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
from time import sleep
import cv2
from pathlib import Path
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from main_gui import Ui_MainWindow
from configuration import Configuration
# from configuration_editor import ConfigurationEditor
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

        # Connect main GUI events with method invocations.
        self.ui.buttonLoadBW.clicked.connect(self.load_bw_image)
        self.ui.buttonLoadColor.clicked.connect(self.load_color_image)
        self.ui.buttonSetConfigParams.clicked.connect(self.edit_configuration)
        self.ui.buttonSaveRegisteredColorImage.clicked.connect(self.save_registered_image)

        # Set the path to the home directory:
        self.home = str(Path.home())

        # Initialize status variables
        self.status_list = [False, False, False, False, False, False]
        self.status_pointer = {"initialized": 0,
                               "bw_loaded": 1,
                               "color_loaded": 2,
                               "rigid_transformed": 3,
                               "optical_flow_computed": 4,
                               "results_written": 5}

        self.radio_buttons = [self.ui.radioShowBW,                   # 0
                              self.ui.radioShowColorOrig,            # 1
                              self.ui.radioShowColorRigidTransform,  # 2
                              self.ui.radioShowMatches,              # 3
                              self.ui.radioShowColorOptFlow]         # 4


        self.max_button = [0, 1, 2, 4, 5, 5]

        # Read in or (if no config file is found) create all configuration parameters. If a new
        # configuration has been created, write it to disk.
        self.configuration = Configuration()

        # Write the program version into the window title.
        self.setWindowTitle(self.configuration.version)

        # Start the workflow thread. It controls the computations and control of external devices.
        # By decoupling those activities from the main thread, the GUI is kept from freezing during
        # long-running activities.
        self.workflow = Workflow(self)
        sleep(self.configuration.wait_for_workflow_initialization)

        # The workflow thread sends signals when a task is finished. Connect those signals with
        # the appropriate GUI activity.
        self.workflow.set_status_signal.connect(self.set_status)

        # Reset downstream status flags.
        self.set_status(0)

    def edit_configuration(self):
        """
        This method is invoked with the "Set configuration parameters" GUI button. Open the
        configuration editor. If the configuration is changed, set the flag which tells the
        workflow thread to repeat the alignment process.

        :return: -
        """

        # editor = ConfigurationEditor(self.configuration)
        editor = None
        editor.exec_()
        if editor.configuration_changed:
            # If parameters have changed, a new alignment has to be computed. If both images are
            # available, set process status to 2. This triggers the computation in the workflow
            # thread.
            if self.current_status > 2:
                self.set_status(2)

    def load_bw_image(self):
        """
        Load the B/W reference image from a file. Keep it together with a grayscale version.

        :return: -
        """

        refFilename = "Images/2018-03-24_20-00MEZ_Mond.jpg"
        print("Reading reference image : ", refFilename)
        self.image_reference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
        self.image_reference_gray = cv2.cvtColor(self.image_reference, cv2.COLOR_BGR2GRAY)
        self.set_status(1)

        # try:
        #     self.image_reference = self.load_image("Load B/W reference image")
        #     self.image_reference_gray = cv2.cvtColor(self.image_reference, cv2.COLOR_BGR2GRAY)
        #     self.set_status(1)
        # except:
        #     self.image_reference = None
        #     self.image_reference_gray = None
        #     self.set_status(0)


    def load_color_image(self):
        """
        Load the color target image from a file. Keep it together with a grayscale version.

        :return: -
        """

        imFilename = "Images/2018-03-24_21-01MEZ_Mond.jpg"
        print("Reading image to align : ", imFilename)
        self.image_target = cv2.imread(imFilename, cv2.IMREAD_COLOR)
        self.image_target_gray = cv2.cvtColor(self.image_target, cv2.COLOR_BGR2GRAY)
        self.set_status(2)

        # try:
        #     self.image_target = self.load_image("Load color image to be registered")
        #     self.image_target_gray = cv2.cvtColor(self.image_target, cv2.COLOR_BGR2GRAY)
        #     self.set_status(2)
        # except:
        #     self.image_target = None
        #     self.image_target_gray = None
        #     self.set_status(1)

    def load_image(self, message, color=False):
        """
        Read an image from a file. Convert it to color mode if optional parameter is set to True.

        :param color: If True, convert image to color mode.
        :return: Numpy array with image data
        """

        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getOpenFileName(self, message, self.home,
                                                  "Images (*.tif *.tiff *.png *.xpm *.jpg)",
                                                  options=options)
        if filename[0] == '':
            raise Exception("File dialog aborted")
        if color:
            return cv2.imread(filename[0], cv2.IMREAD_COLOR)
        else:
            return cv2.imread(filename[0])

    def save_registered_image(self):
        cv2.imwrite("Images\image_reference_gray", self.image_reference_gray)
        cv2.imwrite("D:\SW-Development\Python\LRGB_Registration\Source\Images\image_rigid_transformed_gray", self.image_rigid_transformed_gray)
        # Todo: insert File chooser to get filename, and store the image!
        # cv2.imwrite(str(filename), image)
        self.set_status(5)

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
        self.status_list[status+1 :] = [False] * (len(self.status_list)-status-1)

        # Enable radio buttons which can be used at this point:
        for button in self.radio_buttons[:self.max_button[status]]:
            button.setEnabled(True)
        # Disable the radio buttons for showing images which do not exist at this point.
        for button in self.radio_buttons[self.max_button[status]:]:
            button.setEnabled(False)

        # Update the status bar.
        self.set_statusbar()

        # If both images are loaded and parameters are set, start the computation.
        if status == 2:
            # Tell the workflow thread to compute a new alignment.
            self.workflow.compute_alignment_flag = True

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
        if self.status_list[self.status_pointer["rigid_transformed"]]:
            status_text += ", rigid transformation computed"
        if self.status_list[self.status_pointer["optical_flow_computed"]]:
            status_text += ", images pixel-wise aligned"

        # Tell if results are written to disk.
        if self.status_list[self.status_pointer["results_written"]]:
            status_text += ", results written to disk"

        # Write the complete message to the status bar.
        self.ui.statusbar.showMessage(status_text)

    def closeEvent(self, evnt):
        """
        This event is triggered when the user closes the main window by clicking on the cross in
        the window corner.

        :param evnt: event object
        :return: -
        """

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
