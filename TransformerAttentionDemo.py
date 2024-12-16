import numpy as np
import math


# Q: Vector(Linear layer output) related with what we encode(output, it can be output of encoder layer or decoder layer)
# K: Vector(Linear layer output) related with what we use as input to output.
# V: Learned vector(Linear layer output) as a result of calculations, related with input
# ** source: https://medium.com/analytics-vidhya/understanding-q-k-v-in-transformer-self-attention-9a5eddaa5960
# ** source: https://www.youtube.com/watch?v=QCJQG4DuHT0

# L: length of input sequence
# d_k: dimension of vector "k"
# d_v: dimension of vecor "v"
L, d_k, d_v = 4, 8, 8
q = np.random.randn(L,d_k)
print("q\n", q)
k = np.random.randn(L, d_k)
print("\nk\n", k)
v = np.random.randn(L, d_v)
print("\nv\n", v)

scaled = np.matmul(q, k.T) / math.sqrt(d_k)
print("\nsquared\n", scaled)

mask = np.tril(np.ones((L,L)))
print("\nmask:\n", mask)
mask[mask == 0] = -np.inf
mask[mask == 1] = 0
print("\nmask:\n", mask)

print("\nscaled + mask\n", scaled + mask)

def softmax(x):
    return np.exp(x).T / np.sum(np.exp(x), axis=-1).T

attention = softmax(scaled + mask)
attention = attention.T
print("\nattention\n", attention)
