from fenics import *
import mshr
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri


alpha = 1
domain = mshr.Circle(Point(0.,0.),1.0,60)
mesh = mshr.generate_mesh(domain, 60)

V = FunctionSpace(mesh,'P',1)
u_D = Expression('3 + 2*x[0]*x[0] + 3*x[1]*x[1]',degree=2)

def boundary1(x,on_boundary):
    if on_boundary:
        if x[1]>=0:
            return True
        else:
            return False
    else:
        return False

g = Expression('4*x[0]*x[0]+6*x[1]*x[1]',degree=2)

bc = DirichletBC(V,u_D,boundary1)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('-10 + alpha*(3+2*x[0]*x[0]+3*x[1]*x[1])',degree=2,alpha=alpha)#alpha = 1
a = dot(grad(u),grad(v))*dx + u*v*dx
L = f*v*dx + g*v*ds
u = Function(V)
solve(a == L,u,bc)

#errors

error_L2 = errornorm(u_D, u, 'L2')
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))
print("L2_norm = " + str(error_L2))
print("max_norm = " + str(error_C))

#graph 1
n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1],triangles)

fig1 = plt.figure(1)
zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.savefig('u.png')

#graph 2
fig2 = plt.figure(2)
zfaces2 = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.savefig('u_D.png')
