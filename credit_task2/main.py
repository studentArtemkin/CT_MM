import multiprocessing
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
import Verlet

Body = namedtuple("Body", "mass position velocity")
G = 6.67408 * numpy.power(10.0,-11)







#y: r_0, r_1, ... , r_N, v_0, v_1, ... , v_N
def Equations(y,t,N,dim,mass):
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


def formData(N,dim):
    data = []
    for i in range(N):
        data.append(Body(10.0*numpy.random.rand(1),20.0*numpy.random.rand(dim)-10,20.0*numpy.random.rand(dim)-10))
    return data


def test(data,N,dim):#data is an array of bodies
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
    t = linspace(0,10,101)
    sol = odeint(Equations,y0,t,args=(N,dim,masses))
    return sol


#Verlet method
def solve1(data, N, dim):
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

    sol = numpy.zeros((101,N*dim*2))
    t = linspace(0,10,101)
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


#multithreading

def threading_task(sol,y0,masses,a,a_old, dist_3,N,dim,t,dt, dedicated_id0, dedicated_id1, barrier_list):
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





def solve2(data, N, dim, NofThreads):
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

    sol = numpy.zeros((101,N*dim*2))
    t = linspace(0,10,101)
    dt = t[1]
    a_old = numpy.zeros(N*dim)
    a = numpy.zeros(N*dim)
    dist_3 = numpy.zeros((N,N))

    
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
        thread_list.append(threading.Thread(target=threading_task, args=(sol,y0,masses,a,a_old,dist_3,N,dim,t,dt,int(evaluation_ids[i,0]),int(evaluation_ids[i,1]+1),barrier_list)))
        thread_list[i].start()
    for thread in thread_list:
        thread.join()
    return sol


#multiprocessing
def multiprocessing_task(sol_m,y0,masses,a_m,a_old_m, dist_3_m,N,dim,t,dt, dedicated_id0, dedicated_id1, barrier_list):
    arr0 = numpy.frombuffer(sol_m.get_obj())
    sol = arr0.reshape((101,N*dim*2))
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


def solve3(data, N, dim, NofThreads):
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

    #sol = numpy.zeros((101,N*dim*2))
    t = linspace(0,10,101)
    dt = t[1]
    #a_old = numpy.zeros(N*dim)
    #a = numpy.zeros(N*dim)
    #dist_3 = numpy.zeros((N,N))

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
    sol_m = multiprocessing.Array(c_double,101*N*dim*2)
    arr0 = numpy.frombuffer(sol_m.get_obj())
    sol = arr0.reshape((101,N*dim*2))
    a_m = multiprocessing.Array(c_double,N*dim)
    a_old_m = multiprocessing.Array(c_double,N*dim)
    dist_3_m = multiprocessing.Array(c_double, N*N)
    #myLock = threading.Lock()
    for i in range(evaluation_ids.shape[0]):
        process_list.append(multiprocessing.Process(target=multiprocessing_task, args=(sol_m,y0,masses,a_m,a_old_m,dist_3_m,N,dim,t,dt,int(evaluation_ids[i,0]),int(evaluation_ids[i,1]+1),barrier_list)))
        process_list[i].start()
    for process in process_list:
        process.join()
    return sol



#test
if __name__ == '__main__':
    N = 10
    dim = 1
    NofThreads = 5
    data = formData(N, dim)
    time0 = 0
    time1 = 0
    time2 = 0
    time3 = 0 
    time4 = 0
    tmp_time = time.time()
    solution_test = test(data, N, dim)
    time0 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet = solve1(data, N, dim)
    time1 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_threading = solve2(data, N, dim, NofThreads)
    time2 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_multiproessing = solve3(data, N, dim, NofThreads)
    time3 = time.time() - tmp_time
    tmp_time = time.time()
    solution_Verlet_cython = Verlet.solve4(data, N, dim)
    time4 = time.time() - tmp_time

    print('test: ' + str(solution_test) + '\n')
    print('Verlet: ' + str(solution_Verlet) + '\n')
    print('Verlet_threading: ' + str(solution_Verlet_threading) + '\n')
    print('Verlet_multiprocessing: ' + str(solution_Verlet_multiproessing) + '\n')
    print('Verlet_cython: ' + str(solution_Verlet_cython) + '\n')

    print(str(time0) + ' ' + str(time1) + ' ' + str(time2) + ' ' + str(time3)+ ' ' + str(time4))

    
