import numpy as np

def test():
    a = 0.5
    b = 0.4999999999997
    arr = np.array([[a, a, a], [b, b, b]])
    arr_unq = urt(arr, tol=1e-06)
    print arr, '\n', arr_unq
    return

def urt(data, tol=1e-12, return_index=False, return_inverse=False):

    prec = -np.fix(np.log10(tol))
    d_r = np.fix(data * 10 ** prec) / 10 ** prec + 0.0
    # d_r = np.rint(data * 10 ** prec) / 10 ** prec + 0.0
    # d_r = np.around(data * 10 ** prec) / 10 ** prec + 0.0
    b = np.ascontiguousarray(d_r).view(np.dtype((np.void, d_r.dtype.itemsize * d_r.shape[1])))
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)

    ret_arr = data[ia, :]
    if not return_index and not return_inverse:
        return ret_arr
    else:
        if return_index and return_inverse:
            return ret_arr, ia, ic
        elif return_index:
            return ret_arr, ia
        elif return_inverse:
            return ret_arr, ic

    # if not return_index and not return_inverse:
    #     return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1])
    # else:
    #     if return_index and return_inverse:
    #         return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1]), ia, ic
    #     elif return_index:
    #         return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1]), ia
    #     elif return_inverse:
    #         return np.unique(b).view(d_r.dtype).reshape(-1, d_r.shape[1]), ic
# -----------------------------------------------------------------------------------------------------------

test()