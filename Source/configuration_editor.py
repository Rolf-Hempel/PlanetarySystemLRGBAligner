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

from PyQt5 import QtWidgets
from parameter_configuration import Ui_ConfigurationDialog

class ConfigurationEditor(QtWidgets.QDialog, Ui_ConfigurationDialog):
    """
    Update the parameters used by MoonPanoramaMaker which are stored in the configuration object.
    The interaction with the user is through the ConfigurationDialog class.

    """

    def __init__(self, configuration, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.configuration = configuration
        self.configuration_changed = False

    def accept(self):
        """
        If the OK button is clicked and the configuration has been changed, test all parameters for
        validity. In case an out-of-bound value is entered, open an error correction dialog window.

        :return: -
        """

        if self.configuration_changed:
            pass

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
