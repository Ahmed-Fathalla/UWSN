import numpy as np

def calc_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def random_float(number, x_boundaries, y_boundaries):
    a = (np.random.random((number,1))*(x_boundaries[1]-x_boundaries[0])) + x_boundaries[0]
    b = (np.random.random((number,1))*(y_boundaries[1]-y_boundaries[0])) + y_boundaries[0]
    return np.hstack([a,b])