import numpy as np
import math
import os
import inspect
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GBpy
from GBpy import lattice as GBl
from GBpy import find_csl_dsc as GBfcd
from GBpy import integer_manipulations as GBim
from GBpy import tools as GBt
from GBpy import bp_basis as GBb2
import five_param_fz as fpf
import map_func as mf

def pick_uni_bpn(num, sigma_val, lat_type, bound=10, plot_sw=False):

    ### Creating an instance of lattice class
    elem = GBl.Lattice(lat_type)
    ### Getting the primitive lattice in orthogonal frame
    l_g_go = elem.l_g_go
    ### Extracting the sigma misorientation from the pickle file
    ### Misorientation is in the primitive frame of associated lattice
    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path =  gb_dir + '/pkl_files/cF_Id_csl_common_rotations.pkl'
    pkl_content = pickle.load(open(pkl_path))
    pub_out = []
    for i in range(len(pkl_content[str(sigma_val)]['N'])):
        sig_mis_N = pkl_content[str(sigma_val)]['N'][i]
        sig_mis_D = pkl_content[str(sigma_val)]['D'][i]
        sig_mis_g = sig_mis_N/sig_mis_D

        sig_mis_go = np.dot(np.dot(l_g_go, sig_mis_g), np.linalg.inv(l_g_go)).reshape(1, 3, 3)[0]
        ### Getting the csl basis in primitive frame
        l_csl_g, l_dsc_g = GBfcd.find_csl_dsc(l_g_go, sig_mis_g)
        ### Converting the csl basis to orthogonal frame
        l_csl_go = np.dot(l_g_go, l_csl_g)
        ### reciprocal csl basis in po frame
        l_rcsl_go = GBfcd.reciprocal_mat(l_csl_go)

        mt_rcsl_go = np.dot(l_rcsl_go.transpose(), l_rcsl_go)

        bp_fz, bp_symm_grp, symm_grp_ax, cube_grid, gs = est_bound(bound, mt_rcsl_go, l_rcsl_go, sig_mis_go, num)


        bpn_sphr = fz2sphr(bp_fz, bp_symm_grp, symm_grp_ax)
        bpn = csl_area_sort(bpn_sphr, l_rcsl_go, mt_rcsl_go)
        bpn_grid = bpn_in_grid(bpn, cube_grid, gs, l_rcsl_go, mt_rcsl_go)

        bpn_grid_fz, _, _ = fpf.five_param_fz(sig_mis_go, bpn_grid)
        bpn_grid_fz = GBt.unique_rows_tol(bpn_grid_fz, tol=1e-06)
        bpn_sort, hkl_sort = csl_area_sort(bpn_grid_fz, l_rcsl_go, mt_rcsl_go, return_hkl=True)

        #### Preparing to pickle the contents
        num_hkl = len(hkl_sort)
        print num_hkl, '\n'
        hkl_save = np.hstack((np.arange(1, num_hkl+1, 1).reshape(num_hkl, 1), hkl_sort))
        bpn_save = np.hstack((np.arange(1, num_hkl+1, 1).reshape(num_hkl, 1), bpn_sort))
        mis_id = 'Sig_' + str(sigma_val) + '_' + str(i)
        symm_ax = np.dot(np.linalg.inv(l_g_go), symm_grp_ax)
        sig_attr = [mis_id, hkl_save, bpn_save, sig_mis_g, bp_symm_grp, symm_ax]

        # pkl_file = mis_id + '.pkl'
        # jar = open(pkl_file, 'wb')
        # pickle.dump(sig_attr, jar)
        # jar.close()

        if plot_sw == True:
            plot_2d(bpn_grid, gs)
            grid_lines_sphr = grid(gs)
            plot_3d(grid_lines_sphr, bpn_grid)
            plot_3d(grid_lines_sphr, bpn)

        pub_out.append(sig_attr)

    # print pub_out, '\n'
    return pub_out
def gen_hkl(bound, mt_tensor):

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
    mil_ind_csl = GBt.unique_rows_tol(mil_ind, tol=1e-06)

    ### Antipodal symmetry (h k l) ~ (-h -k -l)
    return mil_ind_csl

def est_bound(bound, mt_rcsl_go, l_rcsl_go, sig_mis_go, num):
    dum_bpn = np.array([[0, 0, 1]])
    _, bp_symm_grp, _ = fpf.five_param_fz(sig_mis_go, dum_bpn)
    cube_grid, gs = make_grid(num, bp_symm_grp)
    req_bpn = 6 * gs * gs
    flag = True
    while flag:
        mil_ind_csl = gen_hkl(bound, mt_rcsl_go)
        ### Converting the miller indices to normals in po frame
        bpn_go = np.dot(l_rcsl_go, mil_ind_csl.transpose()).transpose()
        ### Finding the boundary plane normals in the FZ using five_param_fz
        bp_fz_norms_go1, bp_symm_grp, symm_grp_ax = fpf.five_param_fz(sig_mis_go, bpn_go)
        bp_fz_norms_go1_unq = GBt.unique_rows_tol(bp_fz_norms_go1, tol=1e-06)


        num_bpn_fz = len(bp_fz_norms_go1_unq)
        est_para = 20
        if bp_symm_grp == 'C_s':
            num_bpn_sp = 2*num_bpn_fz
        elif bp_symm_grp == 'C_2h':
            est_para = 50
            num_bpn_sp = 4*num_bpn_fz
        elif bp_symm_grp == 'D_3d':
            num_bpn_sp = 12*num_bpn_fz
        elif bp_symm_grp == 'D_2h':
            num_bpn_sp = 8*num_bpn_fz
            est_para = 200
        elif bp_symm_grp == 'D_4h':
            num_bpn_sp = 16*num_bpn_fz
        elif bp_symm_grp == 'D_6h':
            num_bpn_sp = 24*num_bpn_fz
        elif bp_symm_grp == 'D_8h':
            num_bpn_sp = 32*num_bpn_fz
        elif bp_symm_grp == 'O_h':
            num_bpn_sp = 48*num_bpn_fz


        # para_1 = len(bp_fz_norms_go1)/num_bpn_fz
        if num_bpn_sp >= est_para*req_bpn:
            flag = False
        else:
            bound = bound * 2
    print bound
    return bp_fz_norms_go1_unq, bp_symm_grp, symm_grp_ax, cube_grid, gs

def csl_area_sort(bpn, l_rcsl_go, mt_cslr_go, return_hkl=False):

    mil_sphr = np.dot(np.linalg.inv(l_rcsl_go), bpn.transpose()).transpose()
    mil_sphr = GBim.int_finder(mil_sphr, tol=1e-06, order='rows')
    # mil_sphr = GBt.unique_rows_tol(mil_sphr, tol=1e-06)
    d_inv_sqr = np.diag(np.dot(np.dot(mil_sphr, mt_cslr_go),mil_sphr.transpose()))
    d_inv = np.sqrt(d_inv_sqr)
    ind_d_inv_sort = np.argsort(d_inv)
    # bpn = GBt.unique_rows_tol(bpn[ind_d_inv_sort], tol=1e-04)
    bpn_sort = bpn[ind_d_inv_sort]
    mil_sort = mil_sphr[ind_d_inv_sort]
    if return_hkl == True:
        return bpn_sort, mil_sort
    else:
        return bpn_sort

def make_grid(n, bp_symm_grp):

    if bp_symm_grp == 'C_s':
        theta = 2 * np.pi
    elif bp_symm_grp == 'C_2h':
        theta = np.pi
    elif bp_symm_grp == 'D_3d':
        theta = np.pi/3
    elif bp_symm_grp == 'D_2h':
        theta = np.pi/2
    elif bp_symm_grp == 'D_4h':
        theta = np.pi/4
    elif bp_symm_grp == 'D_6h':
        theta = np.pi/6
    elif bp_symm_grp == 'D_8h':
        theta = np.pi/8
    elif bp_symm_grp == 'O_h':
        theta = np.pi/12

    g = math.ceil(math.sqrt((np.pi*2/3.0)*(n/theta)))
    print g
    beta = math.sqrt(np.pi/6)
    gp = np.linspace(-beta + beta/g,beta - beta/g, g)
    grid = np.transpose([np.tile(gp, len(gp)), np.repeat(gp, len(gp))])

    return grid, g



def bpn_in_grid(bpn, grid, n, l_rcsl_go, mt_rcsl_go, tol=1e-06):

    bpn_c = mf.sphr2cube_2d(bpn)
    beta = math.sqrt(np.pi/6)

    cond1 = abs(bpn_c[:, 0] - beta) < tol
    bpn1 = np.copy(bpn_c[cond1]);bpn11 = bpn1[:, [1, 2]]; bpn_c = np.delete(bpn_c, np.where(cond1)[0], 0)

    cond2 = abs(bpn_c[:, 0] + beta) < tol
    bpn2 = np.copy(bpn_c[cond2]);bpn22 = bpn2[:, [1, 2]]; bpn_c = np.delete(bpn_c, np.where(cond2)[0], 0)

    cond3 = abs(bpn_c[:, 1] - beta) < tol
    bpn3 = np.copy(bpn_c[cond3]);bpn33 = bpn3[:, [2, 0]]; bpn_c = np.delete(bpn_c, np.where(cond3)[0], 0)

    cond4 = abs(bpn_c[:, 1] + beta) < tol
    bpn4 = np.copy(bpn_c[cond4]);bpn44 = bpn4[:, [2, 0]]; bpn_c = np.delete(bpn_c, np.where(cond4)[0], 0)

    cond5 = abs(bpn_c[:, 2] - beta) < tol
    bpn5 = np.copy(bpn_c[cond5]); bpn55 = bpn5[:, [0, 1]]; bpn_c = np.delete(bpn_c, np.where(cond5)[0], 0)

    cond6 = abs(bpn_c[:, 2] + beta) < tol
    bpn6 = np.copy(bpn_c[cond6]); bpn66 = bpn6[:, [0, 1]]; bpn_c = np.delete(bpn_c, np.where(cond6)[0], 0)

    if len(bpn_c) != 0:
        raise "Algorithm is incorrect!"
    bpn_cc = [bpn11, bpn22, bpn33, bpn44, bpn55, bpn66]
    num_bpn = 6*n*n
    bpn_grid = np.zeros((num_bpn, 3))

    for ct1 in range(len(bpn_cc)):
        bpn_l = bpn_cc[ct1]
        for ct2 in range(len(grid)):
            num_bpn  = num_bpn - 1

            g = grid[ct2]; g_x = g[0]; g_y = g[1]
            a_lt_u = (g_x + beta/n)
            a_lt_l = (g_x - beta/n)
            b_lt_u = (g_y + beta/n)
            b_lt_l = (g_y - beta/n)

            tol1 = 1e-06
            cond1a = (a_lt_l - bpn_l[:, 0]) < tol1
            cond1b = (bpn_l[:, 0] - a_lt_u) < tol1
            cond1 = cond1a & cond1b
            cond2a = (b_lt_l - bpn_l[:, 1]) < tol1
            cond2b = (bpn_l[:, 1] - b_lt_u) < tol1
            cond2 = cond2a & cond2b
            cond = cond1 & cond2

            if np.any(cond):
                if ct1 == 0:
                    bpn_pick, ind = pick_ctr(bpn1, l_rcsl_go, mt_rcsl_go, g)
                    bpn1 = np.delete(bpn1, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick
                elif ct1 ==1:
                    bpn_pick, ind = pick_ctr(bpn2, l_rcsl_go, mt_rcsl_go, g)
                    bpn2 = np.delete(bpn2, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick
                elif ct1 == 2:
                    bpn_pick, ind = pick_ctr(bpn3, l_rcsl_go, mt_rcsl_go, g)
                    bpn3 = np.delete(bpn3, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick
                elif ct1 == 3:
                    bpn_pick, ind = pick_ctr(bpn4, l_rcsl_go, mt_rcsl_go, g)
                    bpn4 = np.delete(bpn4, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick
                elif ct1 == 4:
                    bpn_pick, ind = pick_ctr(bpn5, l_rcsl_go, mt_rcsl_go, g)
                    bpn5 = np.delete(bpn5, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick
                elif ct1 == 5:
                    bpn_pick, ind = pick_ctr(bpn6, l_rcsl_go, mt_rcsl_go, g)
                    bpn6 = np.delete(bpn6, ind, 0)
                    bpn_grid[num_bpn, :] = bpn_pick

                bpn_l = np.delete(bpn_l, ind, 0)
            else:
                raise "Insufficient normals to sample!"

    bpn_grid_sphr = mf.cube2sphr_2d(bpn_grid)
    return bpn_grid_sphr

def grid(gs):

    g = gs+1
    beta = math.sqrt(np.pi/6)
    num = 300

    h_x = np.linspace(-beta, beta, g)
    h_y = np.linspace(-beta, beta, num)
    hl = np.transpose([np.tile(h_x, len(h_y)), np.repeat(h_y, len(h_x))])
    h_z = np.zeros(len(hl)); h_z.fill(beta)
    hln = np.column_stack((hl[:, 0], hl[:, 1], h_z))
    hln = np.concatenate((hln, np.column_stack((hl[:, 0], hl[:, 1], -h_z))))
    hln = np.concatenate((hln, np.column_stack((hl[:, 0], h_z, hl[:, 1]))))
    hln = np.concatenate((hln, np.column_stack((hl[:, 0], -h_z, hl[:, 1]))))
    hln = np.concatenate((hln, np.column_stack((h_z, hl[:, 0], hl[:, 1]))))
    hln = np.concatenate((hln, np.column_stack((-h_z, hl[:, 0], hl[:, 1]))))


    v_x = np.linspace(-beta, beta, num)
    v_y = np.linspace(-beta, beta, g)
    vl = np.transpose([np.tile(v_x, len(v_y)), np.repeat(v_y, len(v_x))])
    vln = np.column_stack((vl[:, 0], vl[:, 1], h_z))
    vln = np.concatenate((vln, np.column_stack((vl[:, 0], vl[:, 1], -h_z))))
    vln = np.concatenate((vln, np.column_stack((vl[:, 0], h_z, vl[:, 1]))))
    vln = np.concatenate((vln, np.column_stack((vl[:, 0], -h_z, vl[:, 1]))))
    vln = np.concatenate((vln, np.column_stack((h_z, vl[:, 0], vl[:, 1]))))
    vln = np.concatenate((vln, np.column_stack((-h_z, vl[:, 0], vl[:, 1]))))


    grid_lines = np.concatenate((hln, vln))
    grid_lines_sphr =  GBt.unique_rows_tol(mf.cube2sphr_2d(grid_lines), tol=1e-06)
    return grid_lines_sphr

def plot_3d(pts1, pts2):

    fig = plt.figure(figsize=(15, 15), facecolor='w')
    plt.margins(0.05)
    ax = fig.add_subplot(111, projection = '3d')
    x1 = pts1[:, 0]; y1 = pts1[:, 1]; z1 = pts1[:, 2]
    x2 = pts2[:, 0]; y2 = pts2[:, 1]; z2 = pts2[:, 2]
    ax.scatter(x1, y1, z1, s=0.005, marker='o', c='b')
    ax.scatter(x2, y2, z2, c='r', marker='^')
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    # plt.savefig('3d_plot.png', dpi=100, bbox_inches='tight')
    return True

def plot_2d(pts_a, g, tol=1e-12):

    # g = 6
    g = g +1
    beta = math.sqrt(np.pi/6)
    num = 1000
    h_x = np.linspace(-beta,beta, g)
    h_y = np.linspace(-beta, beta, num)
    hl = np.transpose([np.tile(h_x, len(h_y)), np.repeat(h_y, len(h_x))])

    v_x = np.linspace(-beta, beta, num)
    v_y = np.linspace(-beta,beta, g)
    vl = np.transpose([np.tile(v_x, len(v_y)), np.repeat(v_y, len(v_x))])

    grid_lines = np.concatenate((hl, vl))
    pts_b = grid_lines



    pts = mf.sphr2cube_2d(pts_a)

    cond1 = abs(pts[:, 0] - beta) < tol
    bpn1 = pts[cond1][:, [1, 2]]; pts = np.delete(pts, np.where(cond1)[0], 0)

    cond2 = abs(pts[:, 0] + beta) < tol
    bpn2 = pts[cond2][:, [1, 2]]; pts = np.delete(pts, np.where(cond2)[0], 0)

    cond3 = abs(pts[:, 1] - beta) < tol
    bpn3 = pts[cond3][:, [2, 0]]; pts = np.delete(pts, np.where(cond3)[0], 0)

    cond4 = abs(pts[:, 1] + beta) < tol
    bpn4 = pts[cond4][:, [2, 0]]; pts = np.delete(pts, np.where(cond4)[0], 0)

    cond5 = abs(pts[:, 2] - beta) < tol
    bpn5 = pts[cond5][:, [0, 1]]; pts = np.delete(pts, np.where(cond5)[0], 0)

    cond6 = abs(pts[:, 2] + beta) < tol
    bpn6 = pts[cond6][:, [0, 1]]; pts = np.delete(pts, np.where(cond6)[0], 0)

    pts_1_l = [bpn1, bpn2, bpn3, bpn4, bpn5, bpn6]


    for pts_1 in (pts_1_l):
        fig = plt.figure()

        plt.plot(pts_b[:, 0], pts_b[:, 1],'o', markerfacecolor='b', markersize=0.5)
        plt.plot(pts_1[:, 0], pts_1[:, 1],'o', markerfacecolor='r', markersize=4)
        plt.show()

    return

def fz2sphr(bpn, bp_symm_grp, symm_grp_ax):
    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path = gb_dir + '/pkl_files/'

    if bp_symm_grp == 'C_s':
        file_name = 'symm_mats_Cs.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'C_2h':
        file_name = 'symm_mats_C2h.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'D_3d':
        file_name = 'symm_mats_D3d.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'D_2h':
        file_name = 'symm_mats_D2h.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'D_4h':
        file_name = 'symm_mats_D4h.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'D_6h':
        file_name = 'symm_mats_D6h.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'D_8h':
        file_name = 'symm_mats_D8h.pkl'
        file_path = pkl_path + file_name

    elif bp_symm_grp == 'O_h':
        file_name = 'symm_mats_Oh.pkl'
        file_path = pkl_path + file_name

    rot_mat = np.linalg.inv(symm_grp_ax)
    bpn_rot = np.dot(rot_mat, bpn.transpose()).transpose()
    symm_mat = pickle.load(open(file_path, 'rb'))
    symm_bpn_rot_gop1 = np.tensordot(symm_mat, bpn_rot.transpose(), 1).transpose((1, 2, 0))
    symm_bpn_go1 = np.tensordot(np.linalg.inv(rot_mat), symm_bpn_rot_gop1, 1).transpose(2, 1, 0)

    dim1, dim2, dim3 = np.shape(symm_bpn_go1)
    symm_bpn_go1 = np.reshape(symm_bpn_go1, ((dim1 * dim2), dim3))
    symm_bpn_go1_unq = GBt.unique_rows_tol(symm_bpn_go1, tol=1e-06)
    return symm_bpn_go1_unq

def pick_ctr(bp, l_rcsl_go, mt_cslr_go, grid_ctr, tol = 1e-06):
    bpn = np.copy(bp)
    if len(bpn) == 1:
        return bpn, np.array([])
    bpn_sphr = mf.cube2sphr_2d(bpn)
    mil_sphr = np.dot(np.linalg.inv(l_rcsl_go), bpn_sphr.transpose()).transpose()
    mil_sphr = GBim.int_finder(mil_sphr, tol=1e-06, order='rows')
    d_inv_sqr = np.diag(np.dot(np.dot(mil_sphr, mt_cslr_go), mil_sphr.transpose()))

    d_inv_sqr_min = np.min(d_inv_sqr)
    cond = np.abs(d_inv_sqr-d_inv_sqr_min) <= tol
    # ind1 = np.where(cond)
    bpn_min = bpn[cond]
    num_bpn_min = len(bpn_min)
    d_ctr = np.zeros(num_bpn_min)
    for ct1 in range(num_bpn_min):
        pt = bpn_min[ct1]
        d_ctr[ct1] = np.sqrt((grid_ctr[0] - pt[0])**2 + (grid_ctr[1] - pt[1])**2)
    ind = np.argsort(d_ctr)
    bpn_pick = bpn_min[ind][0]

    ind_r = np.where((bpn[:, 0] == bpn_pick[0]) & (bpn[:, 1] == bpn_pick[1]) & (bpn[:, 2] == bpn_pick[2]))[0]

    return bpn_pick, ind_r

# sig_l = [83]
# # sig_l = list(np.arange(81, 150, 2))
# sig_attr_l = []
# for sig_val in sig_l:
#     print '\n\n', 'SIGMA is', sig_val
#     sig_attr = pick_uni_bpn(16, sig_val, 'cF_Id')
#     sig_attr_l.append(sig_attr)

# pkl_file = 'Sig_81_149.pkl'
# jar = open(pkl_file, 'wb')
# pickle.dump(sig_attr_l, jar)
# jar.close()