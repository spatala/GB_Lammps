import numpy as np
import math
import GBpy.quaternion as GBq
import GBpy.tools as GBt
import GBpy.integer_manipulations as GBim
import scipy.io as sio
import pickle
import os
from misorient_fz_432 import misorient_fz_432

def symm_equi_reps_Oh(r_go1togo2_go1, bpn_go1):


    tol_1 = 1e-04

    bpn_go1 = bpn_go1/np.linalg.norm(bpn_go1)
    # bpn_go1 = GBim.int_finder(bpn_go1)
    misquat = GBq.mat2quat(r_go1togo2_go1)[:-1]
    fz_quat = misorient_fz_432(misquat)
    fz_mat = GBq.quat2mat(GBq.Quaternion(fz_quat))
    symm_gbs =[]

    gb_mat =[]
    gb_mat.append(r_go1togo2_go1)
    gb_mat.append(bpn_go1)
    gb_mat.append('FZ')
    symm_gbs.append(gb_mat)

    gb_mat =[]
    gb_mat.append(r_go1togo2_go1)

    bpn_go1_laue = -bpn_go1
    gb_mat.append(bpn_go1_laue)
    gb_mat.append('FZ')
    symm_gbs.append(gb_mat)

    # gb_mat =[]
    # gb_mat.append(np.linalg.inv(r_go1togo2_go1))
    # gb_mat.append(np.dot(np.linalg.inv(r_go1togo2_go1), bpn_go1))
    # gb_mat.append('FZ')
    # symm_gbs.append(gb_mat)
    #
    # gb_mat =[]
    # gb_mat.append(np.linalg.inv(r_go1togo2_go1))
    # bpn_go1_laue = -np.dot(np.linalg.inv(r_go1togo2_go1), bpn_go1)
    # gb_mat.append(bpn_go1_laue)
    # gb_mat.append('FZ')
    # symm_gbs.append(gb_mat)


    # main_path = os.getcwd()
    # pkl_path = main_path + '/GBpy/pkl_files/symm_mats_O.pkl'
    # symm_mat_O = pickle.load(open(pkl_path, 'r'))
    # orderptgrp = symm_mat_O.shape[0]
    # symm_gbs = []
    # for ct1 in range(orderptgrp):
    #     # print '\n', ct1
    #     for ct2 in range(orderptgrp):
    #         # print '   ', ct2
    #         r_gop1togop2_gop1 = np.dot(np.dot(np.linalg.inv(symm_mat_O[ct1]), r_go1togo2_go1), symm_mat_O[ct2])
    #         bpn_gop1 = np.dot(np.linalg.inv(symm_mat_O[ct1]), bpn_go1)
    #         # bpn_gop1 = GBim.int_finder(bpn_gop1)
    #         gb_mat = []
    #         gb_mat.append(r_gop1togop2_gop1)
    #         gb_mat.append(bpn_gop1)
    #
    #         t_mat1 = np.dot(r_gop1togop2_gop1, fz_mat.T)
    #         # t_mat1 = t_mat1/np.linalg.norm(t_mat1) ### Npt present in MATLAB code, the next mat 2 vec function fails without
    #         t_ax_ang1 = GBt.vrrotmat2vec(t_mat1); t_ang1 = t_ax_ang1[3]
    #         if t_ang1 < tol_1:
    #             gb_mat.append('FZ')
    #         else:
    #             gb_mat.append('Non-FZ')
    #
    #         symm_gbs.append(gb_mat[:])
    #         gb_mat[1] = -gb_mat[1]
    #         symm_gbs.append(gb_mat[:])
    #
    #         r_gop1togop2_gop1 = np.dot(np.dot(np.linalg.inv(symm_mat_O[ct2]), np.linalg.inv(r_go1togo2_go1)), symm_mat_O[ct1])
    #         bpn_gop1 = np.dot(np.dot(np.linalg.inv(symm_mat_O[ct2]), np.linalg.inv(r_go1togo2_go1)), bpn_go1)
    #         # bpn_gop1 = GBim.int_finder(bpn_gop1)
    #         gb_mat2 =[]
    #         gb_mat2.append(r_gop1togop2_gop1)
    #         gb_mat2.append(bpn_gop1)
    #
    #         t_mat1 = np.dot(r_gop1togop2_gop1, fz_mat.T)
    #         # t_mat1 = t_mat1/np.linalg.norm(t_mat1) ### See last inline comment
    #         t_ax_ang1 = GBt.vrrotmat2vec(t_mat1);t_ang1 = t_ax_ang1[3]
    #         if t_ang1 < tol_1:
    #             gb_mat2.append('FZ')
    #         else:
    #             gb_mat2.append('Non-FZ')
    #         # gb_mat = []
    #         # gb_mat.append(gb_mat)
    #         symm_gbs.append(gb_mat2[:])
    #         gb_mat2[1] = -gb_mat2[1]
    #         symm_gbs.append(gb_mat2[:])
    # dum = 0
    return symm_gbs
# r = np.array([[0.8, 0.6, 0], [0.6, -0.8, 0], [0, 0, -1]])
# bpn = [[3], [1], [0]]
# print symm_equi_reps_Oh(r, bpn)








