import numpy
from collections import namedtuple

class obj_model:
    def __init__(self, filename):
        file = open(filename)
        lines = file.readlines()
        state = 0
        tmp_v = []
        tmp_f = []
        self.groups = []
        Group = namedtuple('Group', 'vertices faces')
        


        for line in lines:
            if(state == 0):
                if(line[0] == 'v'):
                    state = 1
            if(state == 1):
                if(line[0] == 'v'):
                    vertex = numpy.fromstring(line.replace('v',' '), dtype=float, sep=' ')
                    tmp_v.append(vertex)
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
                    self.groups.append(Group(tmp_v.copy(),tmp_f.copy()))
                    tmp_v.clear()
                    tmp_f.clear()
                    state = 0
        
            
        
        




        



m = obj_model('model//model3.obj')

file = open('model//model3.obj')
lines = file.readlines()

for group in m.groups:
    print(group.faces[0])

#l = list(map(float, lines[40].replace('v',' ').split(" "))
#print(m.lines[1][0])