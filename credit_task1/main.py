import sys
import math

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)

import OpenGL.GL as gl

import pywavefront
import numpy
import ctypes
import numpy
from collections import namedtuple



#obj file parsing
class obj_model:
    def __init__(self, filename):
        file = open(filename)
        lines = file.readlines()
        state = 0
        #tmp_v = []
        tmp_f = []
        self.groups = []
        self.vertices = []
        Group = namedtuple('Group', 'faces')
        


        for line in lines:
            if(state == 0):
                if(line[0] == 'v'):
                    state = 1
            if(state == 1):
                if(line[0] == 'v'):
                    vertex = numpy.fromstring(line.replace('v',' '), dtype=float, sep=' ')
                    self.vertices.append(vertex)
                elif(line[0] != 'v'):
                    state = 2
            if(state == 2):
                if(line[0] == 'f'):
                    state = 3
            if(state == 3):
                if(line[0] == 'f'):
                    face = numpy.fromstring(line.replace('f',' '), dtype=int, sep=' ')
                    tmp_f.append(face)
                elif(line[0] != 'f'):
                    self.groups.append(Group(tmp_f.copy()))
                    #tmp_v.clear()
                    tmp_f.clear()
                    state = 0

class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        self.glWidget = GLWidget()

        self.xSlider = self.createSlider()
        self.ySlider = self.createSlider()
        self.zSlider = self.createZoomSlider()

        self.xSlider.valueChanged.connect(self.glWidget.setXRotation)
        self.glWidget.xRotationChanged.connect(self.xSlider.setValue)
        self.ySlider.valueChanged.connect(self.glWidget.setYRotation)
        self.glWidget.yRotationChanged.connect(self.ySlider.setValue)
        self.zSlider.valueChanged.connect(self.glWidget.setZRotation)
        self.glWidget.zRotationChanged.connect(self.zSlider.setValue)
        mainLayout = QHBoxLayout()

        glLayout = QVBoxLayout()
        glLayout.addWidget(self.glWidget)
        glLayout.addWidget(self.xSlider)
        glLayout.addWidget(self.ySlider)
        glLayout.addWidget(self.zSlider)
        mainLayout.addLayout(glLayout)

        self.setLayout(mainLayout)

        self.xSlider.setValue(15 * 16)
        self.ySlider.setValue(345 * 16)
        self.zSlider.setValue(3)

        self.setWindowTitle("Credit Task 1")

    def createSlider(self):
        slider = QSlider(Qt.Horizontal)

        slider.setRange(0, 360 * 16)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setTickPosition(QSlider.TicksRight)

        return slider

    def createZoomSlider(self):
        slider = QSlider(Qt.Horizontal)

        slider.setRange(1, 21)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksRight)

        return slider


class GLWidget(QOpenGLWidget):
    xRotationChanged = pyqtSignal(int)
    yRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.object = 0
        self.xRot = 0
        self.yRot = 0
        #self.zRot = 0
        self.zoom = 0

        self.lastPos = QPoint()

        self.trolltechGreen = QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

    def getOpenglInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.update()

    def setZRotation(self, angle):
        0
        z = self.normalizeAngle(angle)
        if z != self.zoom:
            self.zoom = z
            self.zRotationChanged.emit(z)
            self.update()
        #angle = self.normalizeAngle(angle)
        #if angle != self.zRot:
            #self.zRot = angle
            #self.zRotationChanged.emit(angle)
            #self.update()

    def initializeGL(self):
        print(self.getOpenglInfo())

        self.setClearColor(self.trolltechPurple.darker())
        #self.object = self.makeObject()
        self.object = self.LoadModel()
        #print(str(len(object)))
        gl.glShadeModel(gl.GL_FLAT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def paintGL(self):
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        pos = self.zoom/100
        #print(pos)
        gl.glScaled(pos,pos,0.1)
        gl.glTranslated(0.0, -10.0, -80.0)
        gl.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        gl.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        #gl.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0) #TBR
        gl.glCallList(self.object)

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side,
                           side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dy)
            #self.setZRotation(self.zRot + 8 * dx)

        self.lastPos = event.pos()

    def LoadModel2(self):

        self.scene = pywavefront.Wavefront('model\\model3.obj', collect_faces=True)#TBA: name selection
        #print(str(self.separetion))
        self.model = pymesh.load_mesh('model\\model3.obj')
        

        genList = gl.glGenLists(1)

        gl.glNewList(genList, gl.GL_COMPILE)
        gl.glBegin(gl.GL_TRIANGLES)
        self.setColor(self.trolltechGreen)
        
        for face in self.scene.mesh_list[0].faces:
            for vertx in face:
                gl.glVertex3d (self.scene.vertices[vertx][0], self.scene.vertices[vertx][1], self.scene.vertices[vertx][2])

        gl.glEnd()
        gl.glEndList()


        return genList

    def LoadModel(self):
        self.model = obj_model('model\\model3.obj')#TBA: name selection
        genList = gl.glGenLists(1)

        gl.glNewList(genList, gl.GL_COMPILE)
        gl.glBegin(gl.GL_TRIANGLES)
        c = 0
        for group in self.model.groups:
            self.setColor(QColor.fromCmykF(0.12*c, 0.0, 1.0, 0.0))
            c +=1
            for face in group.faces:
                for vertx in face:
                    gl.glVertex3d(self.model.vertices[vertx-1][0],self.model.vertices[vertx-1][1],self.model.vertices[vertx-1][2])
        
        gl.glEnd()
        gl.glEndList()


        return genList




    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def setClearColor(self, c):
        gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())