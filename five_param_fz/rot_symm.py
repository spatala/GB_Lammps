import numpy as np
import pickle
import inspect
import GBpy
import os
import math
def rot_symm(symm_grp_ax, bp_norms_go1, file_path):
    """
    Returns the symmetrically equivalent boundary plane normals

    Parameters
    ----------
    symm_grp_ax: principle axes for the bicrystal fundamental zone
    * numpy array of size (3 x 3)

    bp_norms_go1: normalized boundary plane normals in the po1 reference frame
    * numpy array of size (n x 3)

    file_path: path to the relevant symmetry operations containing pickle file
    * string

    Returns
    -------
    symm_bpn_go1: symmetrically equivalent boundary plane normals in the po1 reference frame
    *  numpy array of size (m x n x 3); m == order of bicrystal point group symmetry group

    Notes
    ------

    """
    bpn_rot = np.dot(np.linalg.inv(symm_grp_ax), bp_norms_go1.transpose()).transpose()
    symm_mat = pickle.load(open(file_path, 'rb'))
    ### np.dot returns the sum product of the last axis of the first matrix with the second to last axis of the second
    ### advisable to use np.tensordot instead to avoid confusion !!
    symm_bpn_rot_gop1 = np.tensordot(symm_mat, bpn_rot.transpose(), 1).transpose((1, 2, 0))
    symm_bpn_go1 = np.tensordot(symm_grp_ax, symm_bpn_rot_gop1, 1).transpose(2, 1, 0)

    return symm_bpn_go1

def test_rot_symm(sig, r):
    """
    tests the rot_symm method

    Parameters
    ----------
    sig: sigma value of the grain boundary
    * odd integer

    r: input to create r number of unit boundary plane normals
    * integer

    Returns
    --------
    None

    Notes
    ------
    * The unit boundary plane normals are stored in the variable bpn, which is a numpy array of size (r x 3).
    """
    gbpy_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path = gbpy_dir + '/pkl_files/'

    if sig == 3:
        z_g = np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
        x_g = np.array([2/math.sqrt(3), -1/math.sqrt(3), -1/math.sqrt(3)])/math.sqrt(2)
        z_g = z_g/np.linalg.norm(z_g)
        x_g = x_g/np.linalg.norm(x_g)
        y_g = np.cross(z_g,x_g)
        sga = np.column_stack((x_g, y_g, z_g))
        fp = pkl_path + 'symm_mats_D6h.pkl'

    if sig == 5:
        z_g = np.array([0, -1/math.sqrt(2), 1/math.sqrt(2)])
        x_g = np.array([1, 0, 0])
        z_g = z_g/np.linalg.norm(z_g)
        x_g = x_g/np.linalg.norm(x_g)
        y_g = np.cross(z_g,x_g)
        sga = np.column_stack((x_g, y_g, z_g))
        fp = pkl_path + 'symm_mats_D4h.pkl'

    bpn = np.zeros((r, 3))
    for ct1 in range(r):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()

        bpn_rand = np.array([x, y, z])
        if np.all(bpn_rand != np.array([0, 0, 0])):
            bpn_rand = bpn_rand/np.linalg.norm(bpn_rand)
        bpn[ct1, :] = bpn_rand

    symm_bpn = rot_symm(sga, bpn, fp)
    print symm_bpn
    return

test_rot_symm(3, 5)




