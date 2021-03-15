import numpy as np
import time
import matplotlib.pyplot as plt



def generate_matrix(size):
    m =200*(np.random.rand(size,size)-0.5)
    while np.linalg.det(m) == 0:
        m =200*(np.random.rand(size,size)-0.5)
    return m

def solve_SLAE(A,b):

    detA = np.linalg.det(A)
    solution = np.zeros(b.shape[0])
    

    for i in range(b.shape[0]):
        tmp = np.copy(A)
        for j in range(b.shape[0]):
            tmp[j,i] = b[j]
        det_i = np.linalg.det(tmp)
        solution[i] = det_i/detA

    return solution


def build_graph(n_of_dimensions):
    result = np.zeros(n_of_dimensions)
    for d in range(1,n_of_dimensions):
        tmp = np.zeros(5)
        for x in range(5):
            t1 = time.time()
            solve_SLAE(generate_matrix(d),200*(np.random.rand(d)-0.5))
            t2 = time.time()
            tmp[x] = t2 - t1
        result[d] = np.mean(tmp)
    plt.plot(result)
    plt.ylabel('seconds')
    plt.xlabel('dimensions')
    plt.show()
    


build_graph(100)

#solve_SLAE(generate_matrix(2),200*(np.random.rand(2)-0.5))