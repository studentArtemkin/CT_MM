from collections import namedtuple
from libc.stdlib cimport malloc, free
from libc.math cimport pow
import numpy

Body = namedtuple("Body", "mass position velocity")



#@cython.cfunc
def solve4(data, N, dim,G,t):
    cdef int N_c = N
    cdef int dim_c = dim
    cdef double G_c = G
    cdef int t_len_c = t.shape[0]
    cdef int i, j, k, n
    cdef double dt = t[1]
    cdef double tmp 
    cdef double *y0 = <double *> malloc(2*N_c*dim_c*sizeof(double))
    cdef double *masses = <double *> malloc(N*sizeof(double))
    cdef double **sol = <double **> malloc(t_len_c*sizeof(double*))
    for i in range(t_len_c):
        sol[i] = <double *> malloc(2*N_c*dim_c*sizeof(double))
    cdef double *a_old = <double *> malloc(N_c*dim_c*sizeof(double))
    cdef double *a = <double *> malloc(N_c*dim_c*sizeof(double))
    cdef double **dist_3 = <double **> malloc(N_c*sizeof(double*))
    for i in range(N_c):
        dist_3[i] = <double *> malloc(N_c*sizeof(double))
    
    i = 0
    for elem0 in data:
        for elem1 in elem0.position:
            y0[i] = elem1
            i += 1
    for elem0 in data:
        for elem1 in elem0.velocity:
            y0[i] = elem1
            i += 1
    i = 0
    for elem in data:
        masses[i] = elem.mass[0]
        i += 1

    
    #starting acceleration
    for i in range(N_c):
        for j in range(N_c):
            tmp = 0
            for k in range(dim_c):
                tmp += pow(y0[i*dim_c+k]-y0[j*dim_c+k],2)
            tmp = pow(tmp,3/2)
            dist_3[i][j] = tmp#copy??
    for i in range(N_c):
        for k in range(dim_c):
            tmp = 0
            for j in range(N_c):
                if i!=j:
                    tmp += masses[j] * (y0[j*dim_c+k]-y0[i*dim_c+k])
                    if dist_3[j][i] != 0:
                        tmp /= dist_3[j][i]
            tmp *= G_c
            a[i*dim_c + k] = tmp
    #method
    for n in range(t_len_c):
        if n == 0:
            for i in range(2*N_c*dim_c):
                sol[0][i] = y0[i]
        else:
            for i in range(N_c*dim_c):
                a_old[i] = a[i]
            #find distances
            for i in range(N_c):
                for j in range(N_c):
                    tmp = 0
                    for k in range(dim_c):
                        tmp += pow(sol[n-1][i*dim_c+k]-sol[n-1][j*dim_c+k],2)
                    tmp = pow(tmp,3/2)
                    dist_3[i][j] = tmp
            #r
            for i in range(N_c*dim_c):
                sol[n][i] = sol[n-1][i] + sol[n-1][N_c*dim_c+i]*dt + a_old[i]*dt*dt/2
            
            #find acceleration
            for i in range(N_c):
                for k in range(dim_c):
                    tmp = 0
                    for j in range(N_c):
                        if i!=j:
                            tmp += masses[j] * (sol[n][j*dim_c+k] - sol[n][i*dim_c+k])
                            if dist_3[j][i] != 0:
                                tmp /= dist_3[j][i]
                    tmp *= G_c
                    a[i*dim_c+k] = tmp
            #v
            for i in range(N_c*dim_c):
                sol[n][N_c*dim_c+i] = sol[n-1][N_c*dim_c+i] + (a[i]+a_old[i])*dt/2
    
    result = numpy.zeros((t_len_c,N_c*dim_c*2))
    for i in range(t_len_c):
        for j in range(N_c*dim_c*2):
            result[i,j] = sol[i][j]


    #reallocating memory
    for i in range(N_c):
        free(dist_3[i])
    free(dist_3)
    free(a)
    free(a_old)
    for i in range(t_len_c):
        free(sol[i])
    free(sol)
    free(masses)
    free(y0)
    return result

