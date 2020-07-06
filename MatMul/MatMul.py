import numpy as np
import time
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule

mod = SourceModule("""
        __global__ void multiplication(double* A, double* B, int* N, double* C){
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int column = blockIdx.x * blockDim.x + threadIdx.x;
                for(int i = 0; i < N; i++){
                        C[row * N + column] += A[row * N + i] * B[i * N + column];              
                }       
        }
""")

N = 2048
A = np.random.randn(N, N)
B = np.random.randn(N, N)
A_matrix = np.matrix(A)
B_matrix = np.matrix(B)
C = np.zeros((N, N))

block_size = (16, 16, 1)
grid_size = (int((N + block_size[0] - 1) / 2), int((N + block_size[1] - 1) / 2))
mult = mod.get_function("multiplication")

start_cpu = time.time()
res_cpu = A.dot(B)
end_cpu = time.time()

start_gpu = time.time()
mult(driver.In(A), driver.In(B), driver.In(N), driver.Out(C), block = block_size, grid = grid_size)
driver.Context.synchronize()
end_gpu = time.time()

print('GPU: ',end_gpu - start_gpu,'\n')
print('CPU: ',end_cpu - start_cpu)

if np.allclose(C, res_cpu):
        print('Results converge')
else:
        print('Results diverge')