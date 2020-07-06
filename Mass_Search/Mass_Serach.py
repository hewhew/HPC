import numpy as np
from numba import cuda
import time
from random import choice, randint
from string import ascii_letters

@cuda.jit
def search_gpu(R, H, N):
    x, y = cuda.grid(2)
    for k in range(N.shape[0]):
        if x < len(H) and N[k][0] == H[x]:
            R[N[k][1], x-N[k][2]] -= 1
    return R

def search_cpu():
    for i, x in enumerate(H):
        for k in n:
            if x == k[0]:
                R[k[1], i - k[2]] -= 1
    return R

Hn = 512
Nn = 25
N_min = 2
N_max = 3

file_H = open('H.txt', 'w');
file_N = open('N.txt', 'w');

H = ''.join(choice(ascii_letters) for i in range(Hn))
alphabet = set(H)
N = []
for x in range(Nn):
   N.append(''.join(choice(ascii_letters) for i in range(randint(N_min, N_max))))
file_H.write(H);
file_N.write(N);
file_H.close();
file_N.close();

n = []
for x in alphabet:
    i_count = 0
    j_count = 0
    for i in N:
        for j in i:
            if x == j:
                n.append((ord(x), i_count, j_count))
            j_count += 1
        i_count += 1

R = np.zeros((Nn, Hn));
count = 0
for el in N:
    R[count] = len(el)

H = np.array([ord(x) for x in H])
blockspergrid = (16,16)
threadsperblock = (16,16)


R_cuda = R.copy()
start = time.time()
new_R = search_gpu[blockspergrid, threadsperblock](R_cuda, H, n)
end = time.time()
print((end - start) * 1000)
start = time.time()
print(R)
new_R = search_cpu()
end = time.time()
print((end - start) * 1000)

