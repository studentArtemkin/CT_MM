import sys
from fenics import *
import mshr
import numpy as np

import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve0(variant):
    T = 20.0
    num_steps = 20
    dt = T / num_steps
    alpha = 1

    cur_var = int(variant)
    if cur_var == 1:
        h = Expression('3 + 2*x[0]*x[0] + 3*x[1]*x[1] + x[0]*t',degree=2,t=0)
        g = Expression('sqrt(x[0]*x[0]+x[1]*x[1])*(4*pow(cos(atan2(x[1],x[0])),2)+6*pow(sin(atan2(x[1],x[0])),2))+t*cos(atan2(x[1],x[0]))',degree=2,t=0)
        f = Expression('x[0] - alpha*10',degree=1,alpha=alpha,t=0)
    elif cur_var == 2:
        h = Expression('(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2) + t*x[0]*x[1]',degree=2,t=0)
        g = Expression('sqrt(x[0]*x[0]+x[1]*x[1])*(2*pow(cos(atan2(x[1],x[0])),2)+t*sin(2*atan2(x[1],x[0])))',degree=2,t=0)
        f = Expression('x[0]*x[1] - alpha*(4*pow(cos(atan2(x[1],x[0])),2)-2*cos(2*atan2(x[1],x[0])))',degree=2,alpha=alpha,t=0)
    elif cur_var == 3:
        h = Expression('x[0]*x[0]+x[0]*x[1]+x[1]*t',degree=2,t=0)
        g = Expression('2*sqrt(x[0]*x[0]+x[1]*x[1])*pow(cos(atan2(x[1],x[0])),2)+sqrt(x[0]*x[0]+x[1]*x[1])*sin(2*atan2(x[1],x[0])) + t*sin(atan2(x[1],x[0]))',degree=2,t=0)
        f = Expression('x[1] - alpha*2',degree=2,alpha=alpha,t=0)
    else:
        return
    
    domain = mshr.Circle(Point(0.,0.),1.0,60)
    mesh = mshr.generate_mesh(domain, 60)

    V = FunctionSpace(mesh, 'P', 1)
    def boundary1(x,on_boundary):
        if on_boundary:
            if x[1]>=0:
                return True
            else:
                return False
        else:
            return False

    def boundary(x,on_boundary):
        return on_boundary

    
    bc = DirichletBC(V, h, boundary1)
    u_n = interpolate(h,V)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx + alpha*dt*dot(grad(u),grad(v))*dx
    L = (u_n + dt*f)*v*dx + dt*g*v*ds
    u = Function(V)
    t = 0

    zfaces_orig_arr = []
    zfaces_arr = []
    for n in range(num_steps):
        t += dt
        h.t = t
        g.t = t
        f.t = t
        solve(a==L,u,bc)
        u_e = interpolate(h,V)
        error_C = np.abs(u_e.vector().get_local()-u.vector().get_local()).max()
        error_L2 = errornorm(u_e, u, 'L2')
        print('t = ',t,', error_max =  ', error_C,', error_L2 = ', error_L2)
        zfaces_orig_arr.append(np.asarray([u_e(cell.midpoint()) for cell in cells(mesh)]).copy())
        zfaces_arr.append(np.asarray([u(cell.midpoint()) for cell in cells(mesh)]).copy())
        u_n.assign(u)

    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1],triangles)


    #graph1
    fig1 = plt.figure(1)
    plt.tripcolor(triangulation, facecolors=zfaces_orig_arr[0], edgecolors='k')

    def animate(i):
        plt.tripcolor(triangulation, facecolors=zfaces_orig_arr[i], edgecolors='k')

    ani = animation.FuncAnimation(fig1,animate,np.arange(len(zfaces_orig_arr)))
    plt.colorbar()
    ani.save('u_orig'+variant+'.gif')

    #graph2
    fig2 = plt.figure(2)

    plt.tripcolor(triangulation, facecolors=zfaces_arr[0], edgecolors='k')
    

    def animate2(i):
        plt.tripcolor(triangulation, facecolors=zfaces_arr[i], edgecolors='k')

    ani = animation.FuncAnimation(fig2,animate2,np.arange(len(zfaces_arr)))
    plt.colorbar()
    ani.save('u'+variant+'.gif')

    #graph3
    fig3 = plt.figure(3)

    zfaces_diff = []#abs(zfaces_arr - zfaces_orig_arr)
    for i in range(len(zfaces_orig_arr)):
       zfaces_diff.append(abs(zfaces_arr[i]-zfaces_orig_arr[i]))

    plt.tripcolor(triangulation, facecolors=zfaces_diff[0], edgecolors='k')

    def animate3(i):
        plt.tripcolor(triangulation, facecolors=zfaces_diff[i], edgecolors='k')

    ani = animation.FuncAnimation(fig3,animate3,np.arange(len(zfaces_diff)))
    plt.colorbar()
    ani.save('diff'+variant+'.gif')







if __name__ == '__main__':
    solve0(sys.argv[1])