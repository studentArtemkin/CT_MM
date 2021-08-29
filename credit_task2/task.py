import multiprocessing
from operator import pos
import numpy
import random
import time
import threading
from collections import namedtuple
from numpy.core.function_base import linspace
from scipy.integrate import odeint
from multiprocessing import Barrier, Process
from multiprocessing import Pool
from ctypes import *
import pyximport; pyximport.install()
import Verlet
from numba import cuda, float32
import numba
import math

from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sunpy

#import pyopencl as cl

Body = namedtuple("Body", "mass position velocity")
#G = 6.67408 * numpy.power(10.0,-11)



#create N random bodies in N-dim space
def formData(N,dim):
    data = []
    for i in range(N):
        data.append(Body(10.0*numpy.random.rand(1),20.0*numpy.random.rand(dim)-10,20.0*numpy.random.rand(dim)-10))
    return data


#======================================================================
#odeint================================================================
#======================================================================


#used by odeint
#y: r_0, r_1, ... , r_N, v_0, v_1, ... , v_N
def Equations(y,t,N,dim,mass,G):
    totalN = N*dim
    dydt = []
    for i in range(0,totalN):
        dydt.append(y[totalN+i])
    dist_3 = numpy.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tmp = 0
            for k in range(dim):
                tmp += numpy.power(y[i*dim+k]-y[j*dim+k],2)
            tmp = numpy.power(tmp,3/2)
            dist_3[i,j] = tmp.copy()
    for i in range(N):
        for k in range(dim):
            tmp = 0
            for j in range(N):
                if i != j:
                    tmp += (y[j*dim + k] - y[i*dim + k])*mass[j]
                    if dist_3[j,i] != 0:
                        tmp /= dist_3[j,i]
            tmp *= G
            dydt.append(tmp.copy())
    return dydt



def test(data,N,dim,G,t):#data is an array of bodies
    y0 = []
    for elem0 in data:
        for elem1 in elem0.position:
            y0.append(elem1)
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0.append(elem1)
    masses =[]
    for elem in data:
        masses.append(elem.mass[0])

    #G = 6.67408 * numpy.power(10.0,-11)
    sol = odeint(Equations,y0,t,args=(N,dim,masses,G))
    return sol


#======================================================================
#simple Verlet=========================================================
#======================================================================



def solve1(data, N, dim,G,t):#Verlet method
    y0 = []
    for elem0 in data:
        for elem1 in elem0.position:
            y0.append(elem1)
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0.append(elem1)
    masses =[]
    for elem in data:
        masses.append(elem.mass[0])

    sol = numpy.zeros((t.shape[0],N*dim*2))

    dt = t[1]
    a_old = numpy.zeros(N*dim)
    a = numpy.zeros(N*dim)

    #starting acceleration
    dist_3 = numpy.zeros((N,N))
    #find distances
    for i in range(N):
        for j in range(N):
            tmp = 0
            for k in range(dim):
                tmp += numpy.power(y0[i*dim+k]-y0[j*dim+k],2)
            tmp = numpy.power(tmp,3/2)
            dist_3[i,j] = tmp.copy()

    for i in range(N):
        for k in range(dim):
            tmp = 0
            for j in range(N):
                if i!=j:
                    tmp += masses[j] * (y0[j*dim+k]-y0[i*dim+k])
                    if dist_3[j,i] != 0:
                        tmp /=dist_3[j,i]
            tmp *= G
            a[i*dim + k] = tmp            
    #method
    for n,current_t in enumerate(t):
        if n == 0:
            for i in range(sol.shape[1]):
                sol[0][i] = y0[i]
        else:
            a_old = a.copy()
            #find distances
            dist_3 = numpy.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    tmp = 0
                    for k in range(dim):
                        tmp += numpy.power(sol[n-1,i*dim+k]-sol[n-1,j*dim+k],2)
                    tmp = numpy.power(tmp,3/2)
                    dist_3[i,j] = tmp.copy()
            #r
            for i in range(N*dim):
                sol[n,i] = sol[n-1,i] + sol[n-1,N*dim+i]*dt + a_old[i]*dt*dt/2
            
            #find acceleration
            for i in range(N):
                for k in range(dim):
                    tmp = 0
                    for j in range(N):
                        if i!=j:
                            tmp += masses[j] * (sol[n,j*dim+k]-sol[n,i*dim+k])
                            if dist_3[j,i] != 0:
                                tmp /=dist_3[j,i]
                    tmp *= G
                    a[i*dim + k] = tmp
            #v
            for i in range(N*dim):
                sol[n,N*dim+i] = sol[n-1,N*dim+i] + (a[i]+a_old[i])*dt/2 
    return sol

#======================================================================
#multithreading Verlet
#======================================================================


def threading_task(sol,y0,masses,a,a_old, dist_3,N,dim,t,dt, dedicated_id0, dedicated_id1, barrier_list,G):
    count_barrier = 0
    #find distances
    for i in range(dedicated_id0,dedicated_id1):
        for j in range(N):
            tmp = 0
            for k in range(dim):
                tmp += numpy.power(y0[i*dim+k]-y0[j*dim+k],2)
            tmp = numpy.power(tmp,3/2)
            dist_3[i,j] = tmp.copy()
    barrier_list[count_barrier].wait()
    count_barrier += 1
    #find accelerations
    for i in range(dedicated_id0,dedicated_id1):
        for k in range(dim):
            tmp = 0
            for j in range(N):
                if i!=j:
                    tmp += masses[j] * (y0[j*dim+k]-y0[i*dim+k])
                    if dist_3[j,i] != 0:
                        tmp /=dist_3[j,i]
            tmp *= G
            a[i*dim + k] = tmp 
    barrier_list[count_barrier].wait()
    count_barrier += 1
    for n,current_t in enumerate(t):
        if n == 0:
            for i in range(2*dedicated_id0,2*dedicated_id1):
                sol[0][i] = y0[i]
            barrier_list[count_barrier].wait()
            count_barrier += 1
        else:
            a_old = a.copy()
            #find distances
            dist_3 = numpy.zeros((N,N))
            barrier_list[count_barrier].wait()
            count_barrier += 1
            for i in range(dedicated_id0,dedicated_id1):
                for j in range(N):
                    tmp = 0
                    for k in range(dim):
                        tmp += numpy.power(sol[n-1,i*dim+k]-sol[n-1,j*dim+k],2)
                    tmp = numpy.power(tmp,3/2)
                    dist_3[i,j] = tmp.copy()
            barrier_list[count_barrier].wait()
            count_barrier += 1
            #r
            for i in range(dedicated_id0*dim,dedicated_id1*dim):
                sol[n,i] = sol[n-1,i] + sol[n-1,N*dim+i]*dt + a_old[i]*dt*dt/2
            barrier_list[count_barrier].wait()
            count_barrier += 1
            
            #find acceleration
            for i in range(dedicated_id0,dedicated_id1):
                for k in range(dim):
                    tmp = 0
                    for j in range(N):
                        if i!=j:
                            tmp += masses[j] * (sol[n,j*dim+k]-sol[n,i*dim+k])
                            if dist_3[j,i] != 0:
                                tmp /=dist_3[j,i]
                    tmp *= G
                    a[i*dim + k] = tmp
            barrier_list[count_barrier].wait()
            count_barrier += 1
            #v
            for i in range(dedicated_id0*dim,dedicated_id1*dim):
                sol[n,N*dim+i] = sol[n-1,N*dim+i] + (a[i]+a_old[i])*dt/2
            barrier_list[count_barrier].wait()
            count_barrier += 1 





def solve2(data, N, dim, NofThreads,G,t):
    y0 = []
    for elem0 in data:
        for elem1 in elem0.position:
            y0.append(elem1)
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0.append(elem1)
    masses =[]
    for elem in data:
        masses.append(elem.mass[0])

    sol = numpy.zeros((t.shape[0],N*dim*2))

    dt = t[1]
    a_old = numpy.zeros(N*dim)
    a = numpy.zeros(N*dim)
    dist_3 = numpy.zeros((N,N))
    #G = 6.67408 * numpy.power(10.0,-11)

    
    barrier_list = []
    for i in range(NofThreads*(2+6*len(t))):
        barrier_list.append(threading.Barrier(NofThreads))

    evaluation_ids = numpy.zeros((NofThreads,2))
    for i in range(NofThreads):
        evaluation_ids[i,0] = i*int(N/NofThreads)
        if i == NofThreads:
            evaluation_ids[i,1] = int(N-1)
        else:
            evaluation_ids[i,1] = int((i+1)*int(N/NofThreads)-1)

    thread_list = []
    #myLock = threading.Lock()
    for i in range(evaluation_ids.shape[0]):
        thread_list.append(threading.Thread(target=threading_task, args=(sol,y0,masses,a,a_old,dist_3,N,dim,t,dt,int(evaluation_ids[i,0]),int(evaluation_ids[i,1]+1),barrier_list,G)))
        thread_list[i].start()
    for thread in thread_list:
        thread.join()
    return sol

#======================================================================
#multiprocessing Verlet
#======================================================================

def multiprocessing_task(sol_m,y0,masses,a_m,a_old_m, dist_3_m,N,dim,t,dt, dedicated_id0, dedicated_id1, barrier_list,G):#G
    arr0 = numpy.frombuffer(sol_m.get_obj())
    sol = arr0.reshape((t.shape[0],N*dim*2))
    a = numpy.frombuffer(a_m.get_obj())
    a_old = numpy.frombuffer(a_old_m.get_obj())
    arr1 = numpy.frombuffer(dist_3_m.get_obj())
    dist_3 = arr1.reshape((N,N))

    count_barrier = 0
    #find distances
    for i in range(dedicated_id0,dedicated_id1):
        for j in range(N):
            tmp = 0
            for k in range(dim):
                tmp += numpy.power(y0[i*dim+k]-y0[j*dim+k],2)
            tmp = numpy.power(tmp,3/2)
            dist_3[i,j] = tmp.copy()
    barrier_list[count_barrier].wait()
    count_barrier += 1
    #find accelerations
    for i in range(dedicated_id0,dedicated_id1):
        for k in range(dim):
            tmp = 0
            for j in range(N):
                if i!=j:
                    tmp += masses[j] * (y0[j*dim+k]-y0[i*dim+k])
                    if dist_3[j,i] != 0:
                        tmp /=dist_3[j,i]
            tmp *= G
            a[i*dim + k] = tmp 
    barrier_list[count_barrier].wait()
    count_barrier += 1
    for n,current_t in enumerate(t):
        if n == 0:
            for i in range(2*dedicated_id0,2*dedicated_id1):
                sol[0][i] = y0[i]
            barrier_list[count_barrier].wait()
            count_barrier += 1
        else:
            a_old = a.copy()
            #find distances
            dist_3 = numpy.zeros((N,N))
            barrier_list[count_barrier].wait()
            count_barrier += 1
            for i in range(dedicated_id0,dedicated_id1):
                for j in range(N):
                    tmp = 0
                    for k in range(dim):
                        tmp += numpy.power(sol[n-1,i*dim+k]-sol[n-1,j*dim+k],2)
                    tmp = numpy.power(tmp,3/2)
                    dist_3[i,j] = tmp.copy()
            barrier_list[count_barrier].wait()
            count_barrier += 1
            #r
            for i in range(dedicated_id0*dim,dedicated_id1*dim):
                sol[n,i] = sol[n-1,i] + sol[n-1,N*dim+i]*dt + a_old[i]*dt*dt/2
            barrier_list[count_barrier].wait()
            count_barrier += 1
            
            #find acceleration
            for i in range(dedicated_id0,dedicated_id1):
                for k in range(dim):
                    tmp = 0
                    for j in range(N):
                        if i!=j:
                            tmp += masses[j] * (sol[n,j*dim+k]-sol[n,i*dim+k])
                            if dist_3[j,i] != 0:
                                tmp /=dist_3[j,i]
                    tmp *= G
                    a[i*dim + k] = tmp
            barrier_list[count_barrier].wait()
            count_barrier += 1
            #v
            for i in range(dedicated_id0*dim,dedicated_id1*dim):
                sol[n,N*dim+i] = sol[n-1,N*dim+i] + (a[i]+a_old[i])*dt/2
            barrier_list[count_barrier].wait()
            count_barrier += 1 



def solve3(data, N, dim, NofThreads,G,t):
    y0 = []
    for elem0 in data:
        for elem1 in elem0.position:
            y0.append(elem1)
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0.append(elem1)
    masses =[]
    for elem in data:
        masses.append(elem.mass[0])



    dt = t[1]
    #G = 6.67408 * numpy.power(10.0,-11)


    barrier_list = []
    for i in range(NofThreads*(2+6*len(t))):
        barrier_list.append(multiprocessing.Barrier(NofThreads))

    evaluation_ids = numpy.zeros((NofThreads,2))
    for i in range(NofThreads):
        evaluation_ids[i,0] = i*int(N/NofThreads)
        if i == NofThreads:
            evaluation_ids[i,1] = int(N-1)
        else:
            evaluation_ids[i,1] = int((i+1)*int(N/NofThreads)-1)

    process_list = []
    sol_m = multiprocessing.Array(c_double,t.shape[0]*N*dim*2)
    arr0 = numpy.frombuffer(sol_m.get_obj())
    sol = arr0.reshape((t.shape[0],N*dim*2))
    a_m = multiprocessing.Array(c_double,N*dim)
    a_old_m = multiprocessing.Array(c_double,N*dim)
    dist_3_m = multiprocessing.Array(c_double, N*N)
    for i in range(evaluation_ids.shape[0]):
        process_list.append(multiprocessing.Process(target=multiprocessing_task, args=(sol_m,y0,masses,a_m,a_old_m,dist_3_m,N,dim,t,dt,int(evaluation_ids[i,0]),int(evaluation_ids[i,1]+1),barrier_list,G)))
        process_list[i].start()
    for process in process_list:
        process.join()
    return sol

#======================================================================
#CUDA 
#======================================================================

CUDA_shared_matrix_size = (400,400)
#not effective probably
@cuda.jit
def Verlet_CUDA(y0,masses, a, a_old, dist_3, sol, N, dim,dt,N_it,G):#G
    #dist = numba.cuda.shared.array(shape=CUDA_shared_matrix_size, dtype=float32)
    x, y = cuda.grid(2)
    tmp = 0
    if (x<N) and (y<N):
        tmp = 0
        for k in range(dim):
            tmp += math.pow(y0[x*dim + k]-y0[y*dim + k],2)
        tmp = math.pow(tmp,3/2)
        dist_3[x,y] = tmp
    cuda.syncthreads()
    if (x<N) and (y<dim):
        tmp=0
        for k in range(N):
            if(x!=k):
                tmp += masses[k]*(y0[k*dim+y]-y0[x*dim+y])
                if dist_3[x,k] != 0:
                    tmp /= dist_3[x,k]
        a[x*dim+y] = tmp
    cuda.syncthreads()
    if (y<dim) and (x<N):
        a[x*dim + y] *= G
    cuda.syncthreads()
    for n in range(N_it):
        if (n == 0):
            if(y < 2*dim) and (x<N):
                sol[0,x*2*dim+y] = y0[x*2*dim+y]
        else:
            if(y<dim) and (x<N):
                a_old[x*dim+y] = a[x*dim+y]
            cuda.syncthreads()
            if(y<dim) and (x<N):
                a[x*dim+y] = 0
            #find distances
            if (x<N) and (y<N):
                tmp = 0.0
                for k in range(dim):
                    tmp += math.pow(sol[n-1,x*dim + k]-sol[n-1,y*dim + k],2)    
                tmp = math.pow(tmp,3/2)
                dist_3[x,y] = tmp
            cuda.syncthreads()
            #r
            if(x<N) and (y<dim):
                sol[n,x*dim+y] = sol[n-1,x*dim+y] + sol[n-1,N*dim+x*dim+y]*dt + a_old[x*dim+y]*dt*dt/2
            cuda.syncthreads()
            #find acceleration
            if (x<N) and (y<dim):
                tmp=0
                for k in range(N):
                    if(x!=k):
                        tmp += masses[k]*(sol[n,k*dim+y]-sol[n,x*dim+y])
                        if dist_3[x,k] != 0:
                            tmp /= dist_3[x,k]
                a[x*dim+y] = tmp
            cuda.syncthreads()
            if (y<dim) and (x<N):
                a[x*dim + y] *= G
            cuda.syncthreads()
            #v
            if (x<N) and (y<dim):
                sol[n,N*dim+x*dim+y] = sol[n-1,N*dim+x*dim+y] + (a[x*dim+y]+a_old[x*dim+y])*dt/2.0
            cuda.syncthreads()



def solve5(data, N, dim,G,t):
    0
    y0 = []
    for elem0 in data:
        for elem1 in elem0.position:
            y0.append(elem1)
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0.append(elem1)
    masses =[]
    for elem in data:
        masses.append(elem.mass[0])


    dt = t[1]
    N_it = t.shape[0]
    np_y0 = numpy.asarray(y0)
    np_masses = numpy.asarray(masses)
    np_dist_3 = numpy.zeros((N,N))
    np_a = numpy.zeros(N*dim)
    np_old_a = numpy.zeros(N*dim)
    np_sol = numpy.zeros((N_it,N*dim*2))

    gpu = cuda.get_current_device()
    threadsperblock = (int(math.sqrt(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))),int(math.sqrt(math.sqrt(gpu.MAX_THREADS_PER_BLOCK)))) #1024
    blockspergrid_x = math.ceil(np_dist_3.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(np_dist_3.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    Verlet_CUDA[blockspergrid, threadsperblock](np_y0, np_masses,np_a, np_old_a, np_dist_3, np_sol, N, dim, dt, N_it,G)

    return np_sol



def test0():
    N = 10
    dim = 1
    NofThreads = 5
    G = 6.67408 * numpy.power(10.0,-11)
    data = formData(N, dim)
    t = linspace(0,10,101)
    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0 
    time4 = 0
    time5 = 0
    tmp_time = time.time()
    solution_test = test(data, N, dim,G,t)
    time0 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet = solve1(data, N, dim,G,t)
    time1 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_threading = solve2(data, N, dim, NofThreads,G,t)
    time2 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_multiproessing = solve3(data, N, dim, NofThreads,G,t)
    time3 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_cython = Verlet.solve4(data, N, dim,G,t)
    time4 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_CUDA = solve5(data, N, dim,G,t)
    time5 = time.time() - tmp_time

    print('test: ' + str(solution_test) + '\n')
    print('Verlet: ' + str(solution_Verlet) + '\n')
    print('Verlet_threading: ' + str(solution_Verlet_threading) + '\n')
    print('Verlet_multiprocessing: ' + str(solution_Verlet_multiproessing) + '\n')
    print('Verlet_cython: ' + str(solution_Verlet_cython) + '\n')
    print('Verlet_CUDA: ' + str(solution_Verlet_CUDA) + '\n')

    print(str(time0) + ' ' + str(time1) + ' ' + str(time2) + ' ' + str(time3)+ ' ' + str(time4) + ' ' + str(time5))




def test_planets():
    kg2Em = 1.67443e-25#earth mass
    sec2day = 1.15741e-5
    m2Au = 6.68459e-12#Astronomical unit
    G = 6.67408 * numpy.power(10.0,-11) * kg2Em * (sec2day**2)/(m2Au**2)
    obstime = Time('2014-05-15T07:54:00.005')
    planet_list = ['sun','earth', 'venus', 'mars', 'mercury', 'jupiter', 'neptune', 'uranus']
    planet_coord = [get_body_heliographic_stonyhurst(this_planet, time=obstime, include_velocity=True) for this_planet in planet_list]
    planet_masses = [1.988e30*kg2Em, 5.972e24*kg2Em, 4.867e24*kg2Em, 6.417e23*kg2Em, 3.301e23*kg2Em, 1.899e27*kg2Em, 1.024e26*kg2Em, 8.682e25*kg2Em]
    

    myData = []
    for count, planet in enumerate(planet_coord):
        myData.append(Body(numpy.array([planet_masses[count]]),numpy.array([planet.data.x.to('au').value, planet.data.y.to('au').value]),numpy.array([planet.velocity.d_x.to('au/day').value,planet.velocity.d_y.to('au/day').value])))

        
    endtime = 370
    t = numpy.linspace(0, endtime, endtime+1)
    sol0 = test(myData,len(myData),2,G,t)
    sol = solve1(myData,len(myData),2,G,t)
    sol2 = solve2(myData,len(myData),2,5,G,t)
    sol3 = solve3(myData,len(myData),2,5,G,t)
    sol4 = Verlet.solve4(myData,len(myData),2,G,t)
    sol5 = solve5(myData,len(myData),2,G,t)

    error1 = sol - sol0
    error2 = sol2 - sol0
    error3 = sol3 - sol0
    error4 = sol4 - sol0
    error5 = sol5 - sol0

    
    table = plt.figure(1)
    #plt.plot(t,error1.mean(1))
    #plt.plot(t,error2.mean(1))
    #plt.plot(t,error3.mean(1))
    #plt.plot(t,error4.mean(1))
    plt.plot(t,error5.mean(1))

    fig = plt.figure(2)
    ax1 = plt.subplot(1, 1, 1, projection='polar')
    position = []
    kmToAU = 6.68459e-9
    for instance in sol:
        tmp = []
        for i in range(0,16,2):
            tmp.append(numpy.sqrt(numpy.power(instance[i],2)+numpy.power(instance[i+1],2)))
            tmp.append(numpy.arctan2(instance[i+1],instance[i]))
        position.append(tmp)

    def update_plot(value):
        ax1.cla()
        for i,name in zip(range(0,16,2),planet_list):
            plt.polar(position[value][i+1],position[0][i],'o',label=name)
        plt.legend()
        fig.canvas.draw_idle()

    for i,name in zip(range(0,16,2),planet_list):
        plt.polar(position[0][i+1],position[0][i],'o',label=name)

    plt.legend()

    ax2 = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider1 = Slider(ax2,'time(years)', 0,endtime,valstep=1)
    slider1.on_changed(update_plot)
    
    plt.show()




#test
if __name__ == '__main__':
    test_planets()
    #test0()

    
