import numpy as np
import math
import os
import pickle
import GBpy as gbp
import GBpy.tools as GBt
from GBpy import integer_manipulations as int_man

def pick_fz_bpl(bp_norms_go1, bp_symm_grp, symm_grp_ax, x_tol):

    x_g = (symm_grp_ax[:, 0]); y_g = symm_grp_ax[:, 1]; z_g = symm_grp_ax[:, 2]
    # main_path = os.getcwd()
    # pkl_path = main_path + '/GBpy/pkl_files/'

    import inspect
    gb_dir = os.path.dirname(inspect.getfile(gbp))
    pkl_path = gb_dir + '/pkl_files/'

    if bp_symm_grp == 'C_s':
        file_name = 'symm_mats_Cs.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'C_2h':
        file_name = 'symms_mats_C2h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D_3d':
        file_name = 'symm_mats_D3d.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D_2h':
        file_name = 'symm_mats_D2h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D_4h':
        file_name = 'symm_mats_D4h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D_6h':
        file_name = 'symm_mats_D6h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D_8h':
        file_name = 'symm_mats_D8h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'O_h':
        file_name = 'symm_mats_Oh.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)



    ### Axes for Stereographic Projection
    ### Z-axis: Along which the points are projected
    z_g = z_g/np.linalg.norm(z_g); y_g = y_g/np.linalg.norm(y_g); x_g = x_g/np.linalg.norm(x_g)
    rot_mat = np.linalg.inv(np.column_stack((x_g, y_g, z_g)))

    ### Keep unique vectors up to a tolerance and in the D6h fundamental zone !!
    ## This is actually not what you are doing here!
    # t1_vecs = np.dot(symm_bpn_go1, rot_mat.T) # Be careful with transposing the rotation matrix
    t1_vecs = np.dot(rot_mat, symm_bpn_go1.transpose()).transpose()
    # new_col = np.linalg.norm(t1_vecs, axis=1)
    # t1_vecs_norm = np.array([new_col, ]*3).T
    # t1_vecs = np.true_divide(t1_vecs, t1_vecs_norm)

    x = t1_vecs[:, 0]; y = t1_vecs[:, 1]; z = t1_vecs[:, 2];

    if bp_symm_grp == 'C_s':
        cond = (z > -x_tol)
    elif bp_symm_grp == 'C_2h':
        cond = (z > -x_tol) & (y > -x_tol)
    elif bp_symm_grp == 'D_3d':
        cond = ((z > -x_tol) & (x > -x_tol)) & ((x/math.sqrt(3) -abs(y) > -x_tol))
    elif bp_symm_grp == 'D_2h':
        cond = ((z > -x_tol) & (y > -x_tol)) & (x > -x_tol)
    elif bp_symm_grp == 'D_4h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/4)*x -y) > -x_tol)
    elif bp_symm_grp == 'D_6h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/6)*x -y) > -x_tol)
    elif bp_symm_grp == 'D_8h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/8)*x -y) > -x_tol)
    elif bp_symm_grp == 'O_h':
        cond = (((z - x) > -x_tol) & ((x - y) > -x_tol)) & (y > -x_tol)

    print cond

    bp_fz_stereo = t1_vecs[cond, :][0]
    ### Selecting the first point in case of multiple symmetrically equivalent points in the FZ ###
    try:
        num_bpfz = np.shape(bp_fz_stereo)[1]
    except:
        num_bpfz = 1
    if num_bpfz != 1:
        bp_fz_stereo = bp_fz_stereo[0, :]

    # bp_fz_norms_go1 = np.dot(bp_fz_stereo, np.linalg.inv(rot_mat).T)
    bp_fz_norms_go1 = np.dot(np.linalg.inv(rot_mat), bp_fz_stereo.transpose()).transpose()

    return bp_fz_norms_go1, bp_fz_stereo

def rot_symm(symm_grp_ax, bp_norms_go1, file_path):
    bpn_rot = np.dot(np.linalg.inv(symm_grp_ax), bp_norms_go1)
    symm_mat_D_4h = pickle.load(open(file_path, 'r'))
    order_ptgrp = np.shape(symm_mat_D_4h)[0]
    symm_bpn_go1 = []
    for ct1 in range(order_ptgrp):
        sm = symm_mat_D_4h[ct1]
        bpn_rot_gop1 = np.dot(sm, bpn_rot)
        bpn_gop1 = np.dot(symm_grp_ax, bpn_rot_gop1)
        symm_bpn_go1.append(bpn_gop1.T[0])
    symm_bpn_go1 = np.array(symm_bpn_go1)
    return symm_bpn_go1

# bsg = 'D_6h'
# bng = np.array([[0, -0.9487, -0.3162], [0, 0.9487, 0.3162], [0, -0.3162, 0.9487], [0, 0.3162, -0.9487]])
# sga = np.array([[0, 0, 1], [-0.3162, -0.9487, 0], [0.9487, -0.3162, 0]])
# xt = 1e-04
# print pick_fz_bpl(bng, bsg, sga, xt)


