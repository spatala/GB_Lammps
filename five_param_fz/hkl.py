import numpy as np
import GBpy
from GBpy import lattice as GBl
from GBpy import find_csl_dsc as GBfcd
from GBpy import tools as GBt
from GBpy import integer_manipulations as GBim
from gen_miller_indices import gen_miller_ind
from uni_grid import uni_grid
from uni_grid import check
import pickle
import os
import inspect
import math
from five_param_fz import five_param_fz



def hkl(bound, sigma_val, lat_type, g):
    """
    Returns a concatenated list containing the boundary plane attributes (normals, symmetry axes and symmetry group of
    the bicrystal)

    Parameters
    ------------
    mesh_size: the integer size for the mesh grid to create boundary plane normals
    * integer

    sigma_val: the sigma value of the grain boundary planes
    * odd integer

    lat_type: the lattice class symbol for the crystal; allowed values 'cF_Id', 'cI_Id'
     * python string

    Returns
    --------
    A python list containing [bp_fz_norms_go1, bp_symm_grp, symm_grp_ax] as the first element and string 'Single' or
    'Multiple' as the second element.

    Notes
    ------
    * 'Single' is returned for sigma boundaries with one misorientation.
    * 'Multiple' is returned for sigma boundaries with multiple misorientations, such as Sigma 13.

    See Also
    ---------
    * find_bpn
    """
    ### Creating an instance of lattice class
    elem = GBl.Lattice(lat_type)
    ### Getting the primitive lattice in orthogonal frame
    l_g_go = elem.l_g_go

    ### Creating a meshgrid of boundary plane miller indices in CSL lattice

    # mil_ind_csl = gen_miller_ind(mesh_size)


    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path =  gb_dir + '/pkl_files//cF_Id_csl_common_rotations.pkl'
    pkl_content = pickle.load(open(pkl_path))
    ### improve for multiple misorientations per sigma
    if len(pkl_content[str(sigma_val)]["N"]) == 1:
        # print '\n Single Sigma Misorientation'
        sig_mis_N = pkl_content[str(sigma_val)]['N'][0]
        sig_mis_D = pkl_content[str(sigma_val)]['D'][0]
        ### Extracting the sigma misorientation from the pickle file
        ### Misorientation is in the primitive frame of associated lattice
        sig_mis_g = sig_mis_N/sig_mis_D
        ### Converting the misorientation to orthogonal frame/superlattice of the crystal
        ### Done using similarity transformation
        sig_mis_go = np.dot(np.dot(l_g_go, sig_mis_g), np.linalg.inv(l_g_go)).reshape(1,3,3)[0]
        ### Getting the csl basis in primitive frame
        l_csl_g, l_dsc_g = GBfcd.find_csl_dsc(l_g_go, sig_mis_g)
        ### Converting the csl basis to orthogonal frame
        l_csl_go = np.dot(l_g_go, l_csl_g)

        l_cslr_go = GBfcd.reciprocal_mat(l_csl_go)
        mt_tensor = np.dot(l_cslr_go.transpose(), l_cslr_go)
        m = 1.0 * bound

        h = math.ceil(math.sqrt(m/mt_tensor[0][0]))
        k = math.ceil(math.sqrt(m/mt_tensor[1][1]))
        l = math.ceil(math.sqrt(m/mt_tensor[2][2]))
        print h, k, l, '\n'
        x_csl = np.arange(-h, h+1, 1.0)
        y_csl = np.arange(-k, k+1, 1.0)
        z_csl = np.arange(-l, l+1, 1.0)
        num_mi = np.size(x_csl)*np.size(y_csl)*np.size(z_csl)
        xx_csl, yy_csl, zz_csl = np.meshgrid(x_csl, y_csl, z_csl, indexing='xy')
        xx_csl, yy_csl, zz_csl = xx_csl.reshape(1, num_mi)[0], yy_csl.reshape(1, num_mi)[0], zz_csl.reshape(1, num_mi)[0]
        mil_ind = np.column_stack([xx_csl, yy_csl, zz_csl])
        ind = np.where((mil_ind[:, 0] == 0) & (mil_ind[:, 1] == 0) & (mil_ind[:, 2] == 0))[0][0]
        ### deleting (0 0 0)
        mil_ind = np.delete(mil_ind, ind, 0)
        ### finding the unique miller indices
        mil_ind = GBim.int_finder(mil_ind, tol=1e-06, order='rows')
        mil_ind_csl = GBt.unique_rows_tol(mil_ind)
        print np.shape(mil_ind_csl)
        gb_fz = find_bpn(mil_ind_csl, l_csl_go, sig_mis_go, g)
        return [gb_fz, 'Single']

    else:
        gb_fz_mult = []
        for i in range(len(pkl_content[str(sigma_val)]['N'])):
            # print '\n Misorientation Number: ', i+1
            sig_mis_N = pkl_content[str(sigma_val)]['N'][i]
            sig_mis_D = pkl_content[str(sigma_val)]['D'][i]
            sig_mis_g = sig_mis_N/sig_mis_D
            sig_mis_go = np.dot(np.dot(l_g_go, sig_mis_g), np.linalg.inv(l_g_go)).reshape(1,3,3)[0]
            l_csl_g, l_dsc_g  = GBfcd.find_csl_dsc(l_g_go, sig_mis_g)
            l_csl_go = np.dot(l_g_go, l_csl_g)
            gb_fz = find_bpn(mil_ind_csl, l_csl_go, sig_mis_go)
            gb_fz_mult.append(gb_fz)
        return [gb_fz_mult, 'Multiple']

def find_bpn(mil_ind_csl, l_csl_go, sig_mis_go, g):
    """
    Converts the miller indices into boundary plane normals(po1) and calls the five_param_fz method


    """

    bpn_go1 = np.dot(GBfcd.reciprocal_mat(l_csl_go), mil_ind_csl.transpose()).transpose()
    bp_fz_norms_go1, bp_symm_grp, symm_grp_ax = five_param_fz(sig_mis_go, bpn_go1)
    bp_fz_norms_go1 = GBt.unique_rows_tol(bp_fz_norms_go1)
    # bp_fz_norms_go1_ug = uni_grid(bp_fz_norms_go1, bp_symm_grp, symm_grp_ax, g)
    bp_fz_norms_go1 = check(bp_fz_norms_go1, bp_symm_grp, symm_grp_ax)
    norms_grp_ax = [bp_fz_norms_go1, bp_symm_grp, symm_grp_ax]
    # norms_grp_ax = [bp_fz_norms_go1_ug, bp_symm_grp, symm_grp_ax]
    return norms_grp_ax

def get_bpn(r, sig, g):
    """
    Calls the hkl function (acts as a wrapper)

    """

    hkl_data = hkl(r, sig, 'cF_Id', g)

    if hkl_data[1] == 'Single':
        bpn = hkl_data[0][0]
        symm_grp = hkl_data[0][1]
        symm_axes = hkl_data[0][2]

        bpn = GBt.unique_rows_tol(bpn, 1e-08)

        print '\n', 'Boundary plane normals in the fundamental zone (po1 reference frame)\n\n', bpn, '\n'
        print 'Symmetry axes for the fundamental zone\n\n', symm_axes
        return bpn, symm_axes, symm_grp, 'S'
    else:
        bpn_mult = []
        symm_axes_mult = []
        symm_grp_mult = []
        for j in range(len(hkl_data[0])):
            bpn = hkl_data[0][j][0][0]
            symm_grp = hkl_data[0][j][0][1]
            symm_axes = hkl_data[0][j][0][2]
            bpn = GBt.unique_rows_tol(bpn, 1e-08)
            bpn_mult.append(bpn)
            symm_axes_mult.append(symm_axes)
            symm_grp_mult.append(symm_grp)
        print '\n', 'Boundary plane normals in the fundamental zone (po1 reference frame)\n\n', bpn_mult, '\n'
        print 'Symmetry axes for the fundamental zone\n\n', symm_axes_mult
    return np.array(bpn_mult), np.array(symm_axes_mult), np.array(symm_grp_mult), 'M'

def re_pick(bpn_fz, sig_mis):
    z_g = np.array([0, 0, 1])
    x_g = np.array([1, 0, 0])
    z_g = z_g/np.linalg.norm(z_g)
    x_g = x_g/np.linalg.norm(x_g)
    y_g = np.cross(z_g,x_g)
    symm_grp_ax = np.zeros((3, 3))
    symm_grp_ax[:, 0] = x_g; symm_grp_ax[:, 1] = y_g; symm_grp_ax[:, 2] = z_g
    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path = gb_dir + '/pkl_files/'
    file_path = pkl_path + 'symm_mats_Oh.pkl'
    bpn_rot = np.dot(np.linalg.inv(symm_grp_ax), bpn_fz.transpose()).transpose()
    symm_mat = pickle.load(open(file_path, 'rb'))
    symm_bpn_rot_gop1 = np.tensordot(symm_mat, bpn_rot.transpose(), 1).transpose((1, 2, 0))
    symm_bpn_go1 = np.tensordot(symm_grp_ax, symm_bpn_rot_gop1, 1).transpose(2, 1, 0)
    symm_bpn_go1 = np.reshape(symm_bpn_go1, ((np.shape(symm_bpn_go1)[0] *np.shape(symm_bpn_go1)[1]),3))
    bp_fz_norms, _, _ = five_param_fz(sig_mis, symm_bpn_go1)

    return bp_fz_norms

