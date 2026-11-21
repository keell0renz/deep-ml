import numpy as np

def orthogonal_projection(v, L):
    v, L = np.array(v, dtype=float), np.array(L, dtype=float)
    c = (v @ L) / (L @ L)
    return (c * L).tolist()
    
