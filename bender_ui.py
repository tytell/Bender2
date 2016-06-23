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

class Ui_BenderWindow(object):
    def setupUi(self, BenderWindow):
        BenderWindow.setObjectName(_fromUtf8("BenderWindow"))
        BenderWindow.resize(737, 694)
        self.centralwidget = QtGui.QWidget(BenderWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.fileNameLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.fileNameLabel.setFont(font)
        self.fileNameLabel.setObjectName(_fromUtf8("fileNameLabel"))
        self.horizontalLayout.addWidget(self.fileNameLabel)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.goButton = QtGui.QPushButton(self.centralwidget)
        self.goButton.setObjectName(_fromUtf8("goButton"))
        self.horizontalLayout.addWidget(self.goButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalSplitter = QtGui.QSplitter(self.centralwidget)
        self.verticalSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.verticalSplitter.setObjectName(_fromUtf8("verticalSplitter"))
        self.verticalLayoutWidget = QtGui.QWidget(self.verticalSplitter)
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.label_2 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 4, 0, 1, 2)
        self.label_3 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 6, 0, 1, 1)
        self.curFileNumberBox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.curFileNumberBox.setMinimum(1)
        self.curFileNumberBox.setMaximum(10000)
        self.curFileNumberBox.setProperty("value", 1)
        self.curFileNumberBox.setObjectName(_fromUtf8("curFileNumberBox"))
        self.gridLayout.addWidget(self.curFileNumberBox, 6, 1, 1, 1)
        self.fileNamePatternEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.fileNamePatternEdit.setObjectName(_fromUtf8("fileNamePatternEdit"))
        self.gridLayout.addWidget(self.fileNamePatternEdit, 5, 1, 1, 4)
        self.browseOutputPathButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.browseOutputPathButton.setObjectName(_fromUtf8("browseOutputPathButton"))
        self.gridLayout.addWidget(self.browseOutputPathButton, 1, 4, 1, 1)
        self.outputPathEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.outputPathEdit.setObjectName(_fromUtf8("outputPathEdit"))
        self.gridLayout.addWidget(self.outputPathEdit, 1, 1, 1, 3)
        self.restartNumberingButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.restartNumberingButton.setObjectName(_fromUtf8("restartNumberingButton"))
        self.gridLayout.addWidget(self.restartNumberingButton, 6, 2, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 6, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.parameterTreeWidget = ParameterTree(self.verticalLayoutWidget)
        self.parameterTreeWidget.setObjectName(_fromUtf8("parameterTreeWidget"))
        self.parameterTreeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout.addWidget(self.parameterTreeWidget)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.loadParametersButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.loadParametersButton.setObjectName(_fromUtf8("loadParametersButton"))
        self.horizontalLayout_2.addWidget(self.loadParametersButton)
        self.saveParametersButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.saveParametersButton.setObjectName(_fromUtf8("saveParametersButton"))
        self.horizontalLayout_2.addWidget(self.saveParametersButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.plotSplitter = QtGui.QSplitter(self.verticalSplitter)
        self.plotSplitter.setOrientation(QtCore.Qt.Vertical)
        self.plotSplitter.setObjectName(_fromUtf8("plotSplitter"))
        self.plot1Widget = PlotWidget(self.plotSplitter)
        self.plot1Widget.setObjectName(_fromUtf8("plot1Widget"))
        self.plot2Widget = PlotWidget(self.plotSplitter)
        self.plot2Widget.setObjectName(_fromUtf8("plot2Widget"))
        self.verticalLayout_2.addWidget(self.verticalSplitter)
        self.verticalLayout_2.setStretch(1, 100)
        BenderWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(BenderWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        BenderWindow.setStatusBar(self.statusbar)

        self.retranslateUi(BenderWindow)
        QtCore.QMetaObject.connectSlotsByName(BenderWindow)

    def retranslateUi(self, BenderWindow):
        BenderWindow.setWindowTitle(_translate("BenderWindow", "Bender", None))
        self.fileNameLabel.setText(_translate("BenderWindow", "TextLabel", None))
        self.goButton.setText(_translate("BenderWindow", "Go", None))
        self.label.setText(_translate("BenderWindow", "Output path:", None))
        self.label_2.setText(_translate("BenderWindow", "File name pattern:", None))
        self.label_3.setText(_translate("BenderWindow", "Trial:", None))
        self.fileNamePatternEdit.setToolTip(_translate("BenderWindow", "Codes:\n"
"{f}=Frequency\n"
"{a}=Amplitude", None))
        self.browseOutputPathButton.setText(_translate("BenderWindow", "Browse...", None))
        self.restartNumberingButton.setText(_translate("BenderWindow", "Restart", None))
        self.loadParametersButton.setText(_translate("BenderWindow", "Load...", None))
        self.saveParametersButton.setText(_translate("BenderWindow", "Save...", None))

from pyqtgraph import PlotWidget
from pyqtgraph.parametertree import ParameterTree

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    BenderWindow = QtGui.QMainWindow()
    ui = Ui_BenderWindow()
    ui.setupUi(BenderWindow)
    BenderWindow.show()
    sys.exit(app.exec_())

