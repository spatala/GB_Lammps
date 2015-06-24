import numpy as np
import GBpy.lattice as GBl
import GBpy.integer_manipulations as GBim
import GBpy.find_csl_dsc as GBfcd
import GBpy.tools as GBt
import pickle
import os
from five_param_fz import five_param_fz

import GBpy as gbp

def hkl(mesh_size, sigma_val, lat_type):

    ### Creating an instance of lattice class
    elem = GBl.Lattice(lat_type)
    ### Getting the primitive lattice in orthogonal frame
    l_g_go = elem.l_g_go

    ### Creating a meshgrid of boundry plane miller indices in CSL lattice
    r = mesh_size
    x_csl = np.arange(-r, r+1, 1.0)
    y_csl = np.arange(-r, r+1, 1.0)
    z_csl = np.arange(-r, r+1, 1.0)
    num_mi = np.size(x_csl)*np.size(y_csl)*np.size(z_csl)
    # z_csl = np.arange(0, r+1, 1.0)
    # num_mi = ((2*r+1)**2)*(r+1)
    # num_mi = ((2*r+1)**3)

    xx_csl, yy_csl, zz_csl = np.meshgrid(x_csl, y_csl, z_csl, indexing='xy')
    xx_csl, yy_csl, zz_csl = xx_csl.reshape(1, num_mi)[0], yy_csl.reshape(1, num_mi)[0], zz_csl.reshape(1, num_mi)[0]
    mil_ind_csl = np.column_stack([xx_csl, yy_csl, zz_csl])
    ind = np.where((mil_ind_csl[:, 0] == 0) & (mil_ind_csl[:, 1] == 0) & (mil_ind_csl[:, 2] == 0))[0][0]
    mil_ind_csl = np.delete(mil_ind_csl, ind, 0)

    # mil_ind_csl = (np.array([[1, 0, 0], [1, 1, 0], [2, 3, 0], [1, 2, 0], [2, 5, 0], [1, 3, 0], [1, 4, 0], [1, 6, 0], [1, 9, 0], [1, 15, 0], [1, 30, 0], [3, 5, 0], [4, 5, 0], [10, 11, 0], [0, 0, 1]]))
    # n = 0; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # n = 1; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # n = 2; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # n = 3; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # n = 4; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # n = 5; mil_ind_csl = np.vstack((mil_ind_csl, (np.array([[1, 0, 2**n], [1, 1, 2**n], [2, 3, 2**n], [1, 2, 2**n], [2, 5, 2**n], [1, 3, 2**n], [1, 4, 2**n], [1, 6, 2**n], [1, 9, 2**n], [1, 15, 2**n], [1, 32**n, 2**n], [3, 5, 2**n], [4, 5, 2**n], [10, 11, 2**n] ]))))
    # print mil_ind_csl

    import inspect
    gb_dir = os.path.dirname(inspect.getfile(gbp))
    print gb_dir

    # main_path = os.getcwd()

    pkl_path =  gb_dir + '/pkl_files//cF_Id_csl_common_rotations.pkl'
    pkl_content = pickle.load(open(pkl_path))
    ### improve for multiple misorientations per sigma
    if len(pkl_content[str(sigma_val)]["N"]) == 1:
        print '\n Single Sigma Misorientation'
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

        gb_fz = find_bpn(mil_ind_csl, l_csl_go, sig_mis_go)
        return [gb_fz, 'Single']

    else:
        gb_fz_mult = []
        for i in range(len(pkl_content[str(sigma_val)]['N'])):
            print '\n Misorientation Number: ', i+1
            sig_mis_N = pkl_content[str(sigma_val)]['N'][i]
            sig_mis_D = pkl_content[str(sigma_val)]['D'][i]
            sig_mis_g = sig_mis_N/sig_mis_D
            sig_mis_go = np.dot(np.dot(l_g_go, sig_mis_g), np.linalg.inv(l_g_go)).reshape(1,3,3)[0]
            l_csl_g, l_dsc_g  = GBfcd.find_csl_dsc(l_g_go, sig_mis_g)
            l_csl_go = np.dot(l_g_go, l_csl_g)
            gb_fz = find_bpn(mil_ind_csl, l_csl_go, sig_mis_go)
            gb_fz_mult.append(gb_fz)
        return [gb_fz_mult, 'Multiple']

def find_bpn(mil_ind_csl, l_csl_go, sig_mis_go):

    bpn_go1 = []
    gb_fz_FB =[]
    for j in range(np.shape(mil_ind_csl)[0]):
        print '    Boundary Case: ', j+1
        ### Converting the miller indices stored in the meshgrid to boundary plane normals in orthogonalframe/super lattice
        bpn_go1_t = np.dot(GBfcd.reciprocal_mat(l_csl_go), mil_ind_csl[j].reshape(3, 1))
        bpn_go1_t = bpn_go1_t/np.linalg.norm(bpn_go1_t)
        # bpn_go1_t = GBim.int_finder(bpn_go1_t)
        bpn_go1.append(bpn_go1_t)
        gb_fz_FB_dum = five_param_fz(sig_mis_go, bpn_go1_t)
        gb_fz_FB.append(gb_fz_FB_dum)

    return gb_fz_FB

def get_bpn(r, sig):

    hkl_data = hkl(r, sig, 'cF_Id')

    if hkl_data[1] == 'Single':
        bpn = np.zeros((len(hkl_data[0]), 3))
        symm_axes = hkl_data[0][0][3]
        bp_symm_grp = hkl_data[0][0][4]
        for i in range(len(hkl_data[0])):
            bpn[i, :] = np.array(hkl_data[0][i][1])

        bpn = GBt.unique_rows_tol(bpn, 1e-08)

        print '\n', 'Boundary plane normals in the fundamental zone (po1 reference frame)\n\n', bpn, '\n'
        print 'Symmetry axes for the fundamental zone\n\n', symm_axes
        return bpn, symm_axes, bp_symm_grp, 'S'
    else:
        bpn_mult = []
        symm_axes_mult = []
        bp_symm_grp_mult = []
        for j in range(len(hkl_data[0])):
            bpn = np.zeros((len(hkl_data[0][j]), 3))
            symm_axes = hkl_data[0][j][0][3]
            bp_symm_grp = hkl_data[0][j][0][4]
            for i in range(len(hkl_data[0][j])):
                bpn[i, :] = hkl_data[0][j][i][1]

            bpn = GBt.unique_rows_tol(bpn, 1e-08)

            bpn_mult.append(bpn)
            symm_axes_mult.append(symm_axes)
            bp_symm_grp_mult.append((bp_symm_grp))
        print '\n', 'Boundary plane normals in the fundamental zone (po1 reference frame)\n\n', bpn_mult, '\n'
        print 'Symmetry axes for the fundamental zone\n\n', symm_axes_mult
    return np.array(bpn_mult), np.array(symm_axes_mult), np.array(bp_symm_grp_mult), 'M'
# print get_bpn(1, 17)



