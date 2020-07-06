import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit

def bil_pixel(image, a, b):
    scale = 2
    i = int(a/scale)
    j = int(b/scale)
    if i + 1 >= image.shape[0]:
        k = 0
    else:
        k = i + 1
    if j + 1 >= image.shape[1]:
        l = 0
    else: l = j + 1
    a = image[i, j]/255
    b = image[k,j]/255
    c = image[i,l]/255
    d = image[k,l]/255
    return (a + b + c + d)*255*0.25

def bilinear(image):
    p = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.uint32)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i, j] = bil_pixel(image, i, j)
    return p

mod = compiler.SourceModule("""
texture<unsigned int, 2, cudaReadModeElementType> tex;
__global__ void do_work(unsigned int * __restrict__ d_result, const int M1, const int M2, const int N2, const int N1, const float * __restrict__ d_xout, 
const float * __restrict__ d_yout)
{
    const int l = threadIdx.x + blockDim.x * blockIdx.x;
    const int k = threadIdx.y + blockDim.y * blockIdx.y;
    float x = (float(l)/N1)*M1;
    float y = (float(k)/N2)*M2;
    if ((l<N1)&&(k<N2)) { d_result[l*N1 + k] = tex2D(tex, x, y); }

}
""")
bilinear_interpolation = mod.get_function("do_work")

image = cv2.imread('original.jpeg', cv2.IMREAD_GRAYSCALE)
M1, N1 = image.shape
M2 = 2*M1
N2 = 2*N1
result = np.zeros((M2, N2), dtype=np.uint32)
block = (16, 16, 1)
grid = (int(np.ceil(M2/block[0])),int(np.ceil(N2/block[1])))
x = np.array([i for i in range(M2)]*N2)
y = np.array([i for i in range(N2)]*M2)

start = driver.Event()
stop = driver.Event()
start.record()

tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.CLAMP)
tex.set_address_mode(1, driver.address_mode.CLAMP)
driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")

bilinear_interpolation(driver.Out(result), np.int32(M1), np.int32(N1), np.int32(M2), np.int32(N2), driver.In(x), driver.In(y), block=block, grid=grid, texrefs=[tex])
stop.record()
gpu_time = stop.time_since(start)
print(gpu_time)
cv2.imwrite("gpu.jpeg", result.astype(np.uint8))

start = timeit.default_timer()
cpu_result = bilinear(image)
cpu_time = timeit.default_timer() - start
print(cpu_time * 1000)
cv2.imwrite("cpu.jpeg", cpu_result.astype(np.uint8))