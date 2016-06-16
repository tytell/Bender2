# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bender.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Bender2Window(object):
    def setupUi(self, Bender2Window):
        Bender2Window.setObjectName(_fromUtf8("Bender2Window"))
        Bender2Window.resize(923, 627)
        self.verticalLayout_2 = QtGui.QVBoxLayout(Bender2Window)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.parameterTree = ParameterTree(Bender2Window)
        self.parameterTree.setObjectName(_fromUtf8("parameterTree"))
        self.parameterTree.headerItem().setText(0, _fromUtf8("1"))
        self.horizontalLayout_2.addWidget(self.parameterTree)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.plot1Widget = PlotWidget(Bender2Window)
        self.plot1Widget.setObjectName(_fromUtf8("plot1Widget"))
        self.verticalLayout.addWidget(self.plot1Widget)
        self.plot2Widget = PlotWidget(Bender2Window)
        self.plot2Widget.setObjectName(_fromUtf8("plot2Widget"))
        self.verticalLayout.addWidget(self.plot2Widget)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.goButton = QtGui.QPushButton(Bender2Window)
        self.goButton.setObjectName(_fromUtf8("goButton"))
        self.horizontalLayout.addWidget(self.goButton)
        self.doneButton = QtGui.QPushButton(Bender2Window)
        self.doneButton.setObjectName(_fromUtf8("doneButton"))
        self.horizontalLayout.addWidget(self.doneButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(Bender2Window)
        QtCore.QMetaObject.connectSlotsByName(Bender2Window)

    def retranslateUi(self, Bender2Window):
        Bender2Window.setWindowTitle(_translate("Bender2Window", "Bender2", None))
        self.goButton.setText(_translate("Bender2Window", "Go", None))
        self.doneButton.setText(_translate("Bender2Window", "Done", None))

from pyqtgraph import PlotWidget
from pyqtgraph.parametertree import ParameterTree
