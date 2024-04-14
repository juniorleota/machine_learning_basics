import numpy as np

def mat_vect_ops():
    vec = np.array([1,2,3])
    vec_t = vec.T
    vec_scale = vec * 10
    mat = np.array([[1,2,3], [4,5,6]])
    mat_t = mat.T
    mat_scale = mat * 10
    mat_add= mat + mat_scale
    mat_add_scalar = mat + 10
    pass

def rand_and_zero_mats():
    row = 2
    col = 3
    rand_mat = np.random.randn(row, col )
    print(rand_mat)
    rand_vec = np.random.randn(row)
    print(rand_vec)
    zeros_mat = np.zeros((row, col))
    print(zeros_mat)
    zeros_vec = np.zeros(4)
    print(zeros_vec)
    zeros_mat_int = np.zeros((row, col), dtype=int)
    print(zeros_mat_int)

def spec_functions():
    log_0 = np.log10(0)
    log_10 = np.log10(10)
    pass

def powers():
    vec = np.array([[1,2,3], [1,2,3]])
    raised_using_np = np.power(vec, [2])
    print(raised_using_np)
    raised = vec ** 2
    print(raised)


if __name__ == "__main__":
    powers()