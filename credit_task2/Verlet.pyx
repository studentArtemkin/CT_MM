import numpy
from collections import namedtuple
from numpy.core.function_base import linspace

Body = namedtuple("Body", "mass position velocity")
#G = 6.67408 * numpy.power(10.0,-11)

def solve4(data, N, dim,G,t):
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