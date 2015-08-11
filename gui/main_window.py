# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created: Tue Aug 11 00:24:44 2015
#      by: PyQt4 UI code generator 4.11.3
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(464, 355)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.btn_browser = QtGui.QPushButton(self.centralWidget)
        self.btn_browser.setGeometry(QtCore.QRect(180, 260, 113, 32))
        self.btn_browser.setObjectName(_fromUtf8("btn_browser"))
        self.label_image = QtGui.QLabel(self.centralWidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 451, 241))
        self.label_image.setText(_fromUtf8(""))
        self.label_image.setObjectName(_fromUtf8("label_image"))
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 464, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        self.menuMain = QtGui.QMenu(self.menuBar)
        self.menuMain.setObjectName(_fromUtf8("menuMain"))
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtGui.QToolBar(MainWindow)
        self.mainToolBar.setObjectName(_fromUtf8("mainToolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtGui.QStatusBar(MainWindow)
        self.statusBar.setObjectName(_fromUtf8("statusBar"))
        MainWindow.setStatusBar(self.statusBar)
        self.actionTrain_Data = QtGui.QAction(MainWindow)
        self.actionTrain_Data.setObjectName(_fromUtf8("actionTrain_Data"))
        self.menuMain.addAction(self.actionTrain_Data)
        self.menuBar.addAction(self.menuMain.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.btn_browser.setText(_translate("MainWindow", "Browser", None))
        self.menuMain.setTitle(_translate("MainWindow", "Main", None))
        self.actionTrain_Data.setText(_translate("MainWindow", "Train Data", None))

