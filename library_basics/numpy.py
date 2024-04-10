import numpy as np

def data_types():
    vec = np.array([1,2,3])
    vec_t = vec.T
    print(f"{vec}\n{vec_t}")
    mat = np.array([[1,2,3], [4,5,6]])
    mat_t = mat.T
    print(f"{mat}\n{mat_t}")

if __name__ == "__main__":
    data_types()