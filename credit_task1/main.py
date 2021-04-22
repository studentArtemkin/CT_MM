import sys
import math

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget, QVBoxLayout, QPushButton, QFileDialog, 
                             QLineEdit, QLabel)

import OpenGL.GL as gl

import numpy
from scipy.spatial import ConvexHull
from scipy.optimize import fsolve
from scipy.integrate import odeint, ode
from collections import namedtuple

from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import threading
import time



A = 0.00000005

Neighbor = namedtuple('Neighbor', 'eid S_ij lmbda')

class Element:
    def __init__(self, e_id, faces, e, c, S_i, neighbors,Q_R):
        self.e_id = e_id
        self.faces = faces
        self.e = e
        self.c = c
        self.S_i = S_i # area of surface
        self.neighbors = neighbors# list of tuples: (eid, S_ij, lmbda)
        self.temperature = 2*e_id
        self.Q_R = Q_R


class M_model:

    def __init__(self):
        self.e = [0.05, 0.05, 0.05, 0.02, 0.1, 0.01, 0.05, 0.05, 0.05]
        self.c = [900, 900, 900, 1930, 520, 840, 900, 900, 900]
        self.lmbda = [240, 240, 118, 9.7, 10.5, 119, 240, 240]
        self.Q_R = ['0','22+2*math.sin(t/8)','0','0','0','0','0','22+2*math.sin(t/6)','0']
        #parse obj file
        self.InitializeElements()


    def InitializeElements(self):
        file = open('model\\model3.obj')
        lines = file.readlines()
        state = 0
        self.vertecies = []
        face_heap = []
        tmp_faces = []
        self.f_elements = []


        for line in lines:
            if(state == 0):
                if(line[0] == 'v'):
                    state = 1
            if(state == 1):
                if(line[0] == 'v'):
                    vertex = numpy.fromstring(line.replace('v',' '), dtype=float, sep=' ')
                    self.vertecies.append(vertex)
                    #read vertices
                elif(line[0] != 'v'):
                    state = 2
            if(state == 2):
                if(line[0] == 'f'):
                    state = 3
            if(state == 3):
                if(line[0] == 'f'):
                    face = numpy.fromstring(line.replace('f',' '), dtype=int, sep=' ')
                    tmp_faces.append(face)
                    #read faces
                elif(line[0] != 'f'):
                    face_heap.append(tmp_faces.copy())
                    tmp_faces = []
                    #save faces
                    state = 0
        
        list_to_sort = []
        for faces in face_heap:
            tmp = []
            for face in faces:
                for v in face:
                    tmp.append(self.vertecies[v-1][1])
            list_to_sort.append(min(tmp))
        np_list_to_sort = numpy.array(list_to_sort)
        elem_order = numpy.argsort(np_list_to_sort)

        for count,elem_id in enumerate(elem_order):
            #find S_i

            eval_seq = []
            faces = numpy.array(face_heap[elem_id])
            minV = numpy.squeeze(numpy.asarray(faces)).min()-1
            maxV = numpy.squeeze(numpy.asarray(faces)).max()-1
            S_i = ConvexHull(self.vertecies[minV:maxV]).area
            #find neighbors
            #1st
            tmp_neighbors = []
            tmp = []
            for i in range(minV,maxV+1):
                tmp.append(self.vertecies[i][1])
            minY = min(tmp)
            maxY = max(tmp)
            if(count != 0):
                tmp = []
                for i in range(minV,maxV+1):
                    if(self.vertecies[i][1] == minY):
                        tmp.append([self.vertecies[i][0],self.vertecies[i][2]])
                S_ij = ConvexHull(tmp[:]).area
                tmp_neighbors.append(Neighbor(count-1,S_ij,self.lmbda[count-1]))
            #2nd
            if(count != (len(elem_order)-1)):
                tmp = []
                for i in range(minV,maxV+1):
                    if(self.vertecies[i][1] == maxY):
                        tmp.append([self.vertecies[i][0],self.vertecies[i][2]])
                S_ij = ConvexHull(tmp[:]).area
                tmp_neighbors.append(Neighbor(count+1,S_ij,self.lmbda[count]))
            self.f_elements.append(Element(count,faces,self.e[count],self.c[count],S_i,tmp_neighbors,self.Q_R[count]))
        self.findTemp()



    def findTemp(self):
        root = fsolve(self.StationarySystem,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        for count, elem in enumerate(self.f_elements):
            elem.temperature = root[count]

        
    def StationarySystem(self,t):
        result = []
        for elem in self.f_elements:
            tmp = 0
            for neighbor in elem.neighbors:
                tmp-=neighbor.lmbda*neighbor.S_ij*(t[neighbor.eid]-t[elem.e_id])
            tmp -= elem.e*elem.S_i*5.67*pow(t[elem.e_id]/100,4)
            result.append(tmp.copy())
        return result

    def Equation(self,Temp,t):
        result = []
        for count, elem in enumerate(self.f_elements):
            tmp=0
            tmp = -elem.e*elem.S_i*5.67*pow(Temp[elem.e_id],4)
            tmp += A*eval(elem.Q_R)
            for neighbor in elem.neighbors:
                tmp -= neighbor.lmbda*neighbor.S_ij*(Temp[neighbor.eid]-Temp[elem.e_id])
            tmp/=elem.c
            result.append(tmp)
        return result


class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        self.results_file_name = 'results.csv'
        self.NofNodes = 100

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
        

        self.save_coefficients_btn = QPushButton('save coefficients',self)
        self.save_coefficients_btn.clicked.connect(self.save_coefficients_button)
        self.load_coefficients_btn = QPushButton('load coefficients',self)
        self.load_coefficients_btn.clicked.connect(self.load_coefficients_button)
        self.results_file_btn = QPushButton('results file',self)
        self.results_file_btn.clicked.connect(self.results_file_button)
        self.results_graph_btn = QPushButton('graph',self)
        self.results_graph_btn.clicked.connect(self.graph_button)
        self.time_textbox = QLineEdit(self)
        self.onlyDoubleValid = QDoubleValidator(0.0,13.0,2)
        self.time_textbox.setValidator(self.onlyDoubleValid)
        self.time_textbox.setText('13.0')
        self.Start_btn = QPushButton('start',self)
        self.Start_btn.clicked.connect(self.start_button)
        self.label = QLabel(self)
        self.label.setText('')

        menuLayout = QVBoxLayout()
        menuLayout.addWidget(self.save_coefficients_btn)
        menuLayout.addWidget(self.load_coefficients_btn)
        menuLayout.addWidget(self.time_textbox)
        menuLayout.addWidget(self.Start_btn)
        menuLayout.addWidget(self.results_file_btn)
        menuLayout.addWidget(self.results_graph_btn)
        menuLayout.addWidget(self.label)
        menuLayout.addStretch(5)
        menuLayout.setSpacing(20)
        menuWidget = QWidget()
        menuWidget.setLayout(menuLayout)

        #colorbar
        colors = [(1.0,1.0,0.0),(1.0,0.0,0.0)]
        values = [-10,10]
        self.first_graph = 1

        fig = plt.figure( figsize=(2,4) )
        ax = fig.add_axes([0, 0.05, 0.25, 0.9])
        norm = mpl.colors.Normalize(vmin=values[0], vmax=values[1])
        cmap = mpl.colors.LinearSegmentedColormap.from_list("myColorBar",list(zip([0.0,1.0],colors)),N=100)
        cb = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='vertical')
        self.cb_canvas = FigureCanvas(fig)
        #~colorbar


        mainLayout.addWidget(menuWidget)
        mainLayout.addLayout(glLayout)
        mainLayout.addWidget(self.cb_canvas)
        

        self.setLayout(mainLayout)

        self.xSlider.setValue(15 * 16)
        self.ySlider.setValue(345 * 16)
        self.zSlider.setValue(3)

        self.setWindowTitle("Credit Task 1")

        self.coef_file_name = 'coefs.csv'

    def load_coefficients_button(self):
        (self.coef_file_name,info) = QFileDialog.getOpenFileName(self, 'Open file', '',"CSV files (*.csv)")
        if self.coef_file_name:
            with open(self.coef_file_name, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                rows = []
                for row in reader:
                    rows.append(row)
                e = []
                c = []
                lmbda = []
                Q_R = []
                for member in rows[0]:
                    e.append(float(member))
                for member in rows[1]:
                    c.append(float(member))
                for member in rows[2]:
                    lmbda.append(float(member))
                for member in rows[3]:
                    Q_R.append(member.replace('\'','').replace(' ',''))
                self.glWidget.model.e = e.copy()
                self.glWidget.model.c = c.copy()
                self.glWidget.model.lmbda = lmbda.copy()
                self.glWidget.model.Q_R = Q_R.copy()
                self.glWidget.model.InitializeElements()




    def save_coefficients_button(self):
        (self.coef_file_name,info) = QFileDialog.getOpenFileName(self, 'Open file', '',"CSV files (*.csv)")
        if self.coef_file_name:
            with open(self.coef_file_name, 'w', newline='') as current_file:
                e_str = str(self.glWidget.model.e).replace('[','').replace(']','')
                c_str = str(self.glWidget.model.c).replace('[','').replace(']','')
                lmbda_str = str(self.glWidget.model.lmbda).replace('[','').replace(']','')
                Q_R_str = str(self.glWidget.model.Q_R).replace('[','').replace(']','')
                current_file.write(e_str+'\n')
                current_file.write(c_str+'\n')
                current_file.write(lmbda_str+'\n')
                current_file.write(Q_R_str)

    def results_file_button(self):
        (possible_results_file_name,info) = QFileDialog.getOpenFileName(self, 'Open file', '',"CSV files (*.csv)")
        if possible_results_file_name:
            self.results_file_name = possible_results_file_name

    def graph_button(self):
        if self.results_file_name:
            with open(self.results_file_name,newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                rows = []
                for row in reader:
                    rows.append(row)
                
                t_min = float(rows[0][0])
                t_max = float(rows[1][0])
                NoftSteps = int(rows[2][0])
                temp = []
                for i in range(3,len(rows)):
                    tmp=[]
                    for member in rows[i]:
                        tmp.append(float(member))
                    temp.append(tmp)
            
            t = numpy.linspace(t_min,t_max,NoftSteps)
            columns = list(zip(*temp))
            plt1 = plt.figure()
            plt.plot(t,columns[0],label='element 1')
            plt.plot(t,columns[1],label='element 2')
            plt.plot(t,columns[2],label='element 3')
            plt.plot(t,columns[3],label='element 4')
            plt.plot(t,columns[4],label='element 5')
            plt.plot(t,columns[5],label='element 6')
            plt.plot(t,columns[6],label='element 7')
            plt.plot(t,columns[7],label='element 8')
            plt.plot(t,columns[8],label='element 9')
            plt.legend()
            plt.show()
            if self.first_graph:
                plt.close(1)
                self.first_graph=0


            self.cb_canvas.update()
            




    def backgroundSolver(self):
        self.Start_btn.setEnabled(False)
        self.results_file_btn.setEnabled(False)
        self.label.setText('Solving')
        duration = int(self.time_textbox.text())
        self.glWidget.model.findTemp()
        
        temp0=[]
        for elem in self.glWidget.model.f_elements:
            temp0.append(elem.temperature)
        t = numpy.linspace(0,duration,self.NofNodes)
        sol = odeint(self.glWidget.model.Equation,temp0,t,atol=0.1,rtol=0.1,hmin=0.01,hmax=0.5)
        for i in range(self.NofNodes):
            for elem in self.glWidget.model.f_elements:
                elem.temperature = sol[i,elem.e_id]
            self.glWidget.update()
            time.sleep(0.1)
        
        if self.results_file_name:
            with open(self.results_file_name, 'w', newline='') as current_file:
                current_file.write(str(0)+'\n'+str(duration)+'\n'+str(self.NofNodes)+'\n')
                for i in range(self.NofNodes):
                    tmp = []
                    for j in range(9):
                        tmp.append(sol[i,j])
                    temp_str = str(tmp).replace('[','').replace(']','').replace('\n','')
                    current_file.write(temp_str+'\n')
        self.label.setText('Solved')
        self.Start_btn.setEnabled(True)
        self.results_file_btn.setEnabled(True)


    
    def start_button(self):
        solution_thread = threading.Thread(target=self.backgroundSolver,name="Background_Solver")
        solution_thread.start()

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

        self.model = M_model()
        self.xRot = 0
        self.yRot = 0
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

    def initializeGL(self):
        print(self.getOpenglInfo())

        self.setClearColor(self.trolltechPurple.darker())
        gl.glShadeModel(gl.GL_FLAT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def paintGL(self):
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        pos = self.zoom/100
        gl.glScaled(pos,pos,0.1)
        gl.glTranslated(0.0, -10.0, -80.0)
        gl.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        gl.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_TRIANGLES)
        self.setColor(self.trolltechGreen)
        #draw here
        for elem in self.model.f_elements:
            mapedTemp = 232-int((elem.temperature+10)/20*231)
            for face in elem.faces:
                self.setColor(QColor.fromRgb(232,mapedTemp,1))
                for vertx in face:
                    gl.glVertex3d(self.model.vertecies[vertx-1][0],self.model.vertecies[vertx-1][1],self.model.vertecies[vertx-1][2])
        #end draw here
        gl.glEnd()

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
            self.setZRotation(self.zRot + 8 * dx)

        self.lastPos = event.pos()


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