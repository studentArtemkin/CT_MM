import sys
from fenics import *
import mshr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri

def solve0(variant):
    cur_var = int(variant)
    alpha = 1
    if(cur_var==1):
        u_D = Expression('3 + 2*x[0]*x[0] + 3*x[1]*x[1]',degree=2)
        g = Expression('4*sqrt(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2)+6*sqrt(x[0]*x[0] + x[1]*x[1])*pow(sin(atan2(x[1],x[0])),2)',degree=2)
        f = Expression('-10 + alpha*(3+2*x[0]*x[0]+3*x[1]*x[1])',degree=2,alpha=alpha)
    elif(cur_var==2):
        u_D = Expression('(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2)',degree=2)
        g = Expression('2*sqrt(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2)',degree=2)
        f = Expression('-(4*pow(cos(atan2(x[1],x[0])),2)-2*cos(2*atan2(x[1],x[0]))) + alpha*(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2)',degree=2,alpha=alpha)
    elif(cur_var==3):
        u_D = Expression('exp(x[0])+x[0]*x[1]',degree=2)
        g = Expression('exp(x[0])*cos(atan2(x[1],x[0]))+sqrt(x[0]*x[0]+x[1]*x[1])*sin(2*atan2(x[1],x[0]))',degree=2)
        f = Expression('-exp(x[0])+alpha*(exp(x[0])+x[0]*x[1])',degree=2,alpha=alpha)
    else:
        return

    domain = mshr.Circle(Point(0.,0.),1.0,60)
    mesh = mshr.generate_mesh(domain, 60)
    V = FunctionSpace(mesh,'P',1)

    def boundary1(x,on_boundary):
        if on_boundary:
            if x[1]>=0:
                return True
            else:
                return False
        else:
            return False

    bc = DirichletBC(V,u_D,boundary1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u),grad(v))*dx + alpha*u*v*dx
    L = f*v*dx + g*v*ds
    u = Function(V)
    solve(a == L,u,bc)

    #errors

    error_L2 = errornorm(u_D, u, 'L2')
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))
    print('norm_L2 = ' + str(error_L2))
    print('error_C = ' + str(error_C))

    #graph 1
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1],triangles)

    fig1 = plt.figure(1)
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.savefig(str('u'+variant+'.png'))

    #graph 2
    fig2 = plt.figure(2)
    zfaces2 = np.asarray([u_D(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces2, edgecolors='k')
    plt.savefig(str('u_D'+variant+'.png'))

    #difference
    fig3 = plt.figure(3)
    zfaces3 = abs(zfaces-zfaces2)
    plt.tripcolor(triangulation, facecolors=zfaces3, edgecolors='k')
    plt.colorbar()
    plt.savefig(str('difference'+variant+'.png'))




if __name__ == '__main__':
    solve0(sys.argv[1])
