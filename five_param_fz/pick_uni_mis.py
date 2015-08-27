import os
import inspect
import pickle
import numpy as np
import GBpy
from GBpy import lattice as GBl
from GBpy import quaternion as GBq
from GBpy import tools as GBt
import map_func as mf
def pick_uni_mis(lat_type, n=70):
    elem = GBl.Lattice(lat_type)
    l_g_go = elem.l_p_po
    l_go_g = np.linalg.inv(l_g_go)
    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path =  gb_dir + '/pkl_files/symm_mats_O.pkl'
    symm_O = pickle.load(open(pkl_path))
    symm_O_inv = np.linalg.inv(symm_O)
    mis_g, id = extract_mis(99, gb_dir)
    mis_go = np.tensordot(np.tensordot(l_g_go, mis_g.transpose(1, 2, 0), 1).transpose(2, 0, 1), l_go_g, 1)
    mis_go_inv = np.linalg.inv(mis_go)

    num = len(symm_O)*len(symm_O_inv)
    mis_go_symm1 = np.tensordot(np.tensordot(symm_O_inv, mis_go.transpose(1, 2, 0), 1).transpose(3, 0, 1, 2), symm_O.transpose(1, 2, 0), 1).transpose(0, 1, 4, 2, 3)
    mis_go_symm2 = np.tensordot(np.tensordot(symm_O_inv, mis_go_inv.transpose(1, 2, 0), 1).transpose(3, 0, 1, 2), symm_O.transpose(1, 2, 0), 1).transpose(0, 1, 4, 2, 3)
    mis_go_symm = np.concatenate((mis_go_symm1, mis_go_symm2))
    mis_go_symm = mis_go_symm.reshape(len(mis_go_symm)*num, 3, 3)
    id_attr3 = np.tile(np.arange(1, num+1, 1).reshape(num, 1), (len(id), 1))
    id_attr3 = np.vstack([id_attr3, -id_attr3])
    id_symm = np.tile(np.repeat(id, num, axis=0), (2, 1))
    id_symm = np.hstack([id_symm, id_attr3])
    quat_go_symm = GBq.mat2quat(mis_go_symm)
    quat_go_symm_fil = GBq.antipodal(quat_go_symm)
    ### Add code for pi rotations
    quat_t = quat_go_symm_fil.transpose()
    quat_t_unq, ind = GBt.unique_rows_tol(quat_t, tol=1e-06, return_index=True)

    id_unq = id_symm[ind]

    ind_id_sort = np.argsort(id_unq[:, 0])
    id_sort = id_unq[ind_id_sort]

    quat_sort = quat_t_unq[ind_id_sort].transpose()
    ### sort again according to sig vals
    mis_c = mf.sphr2cube_3d(quat_sort)

    ### testing
    # mis_r = mf.cube2sphr_3d(mis_c)

    grid, gce = make_grid(n)
    mis_gr_c, id_g = mis_in_grid(mis_c, id_sort, grid, gce)
    id_g_unq = GBt.unique_rows_tol(id_g[:, [0, 1]], 1e-06)
    ind_id_unq_sort = np.argsort(id_g_unq[:, 0]); id_g_unq = id_g_unq[ind_id_unq_sort]
    mis_pick_ind = mf.set_int_ind(id, id_g_unq)
    mis_pick = mis_g[mis_pick_ind]
    print '\n', len(mis_pick), '\n', id_g_unq
    # mis_gr_quat = mf.cube2sphr_3d(mis_gr_c)
    # mis_gr_go = GBq.quat2mat(mis_gr_quat)
    # mis_gr = np.tensordot(np.tensordot(l_go_g, mis_gr_go.transpose(1, 2, 0), 1).transpose(2, 0, 1), l_g_go, 1)
    # ### convert to g frame
    # ### pick in fz, check the code !!
    # dum = 0
    return [mis_pick, mis_pick_ind]

def extract_mis(num, gb_dir):
    pkl_path = gb_dir + '/pkl_files/cF_Id_csl_common_rotations.pkl'
    pkl_content = pickle.load(open(pkl_path))
    l1 = []; l2 =[]
    for j in range(num):
        sigma_val = 3 + 2*j
        for i in range(len(pkl_content[str(sigma_val)]['N'])):
            sig_mis_N = pkl_content[str(sigma_val)]['N'][i]
            sig_mis_D = pkl_content[str(sigma_val)]['D'][i]
            sig_mis_g = sig_mis_N/sig_mis_D
            l1.append(sig_mis_g)
            l2.append([float(sigma_val), float(i)])
    # mis_arr = np.array(mis_l)
    return np.array(l1), np.array(l2)

def make_grid(n):
    gs = 2*np.floor((n*24)**(1.0/3))
    # gs = 10
    edge_c = np.pi**(2.0/3)
    gce = edge_c/gs

    coor = np.arange(-edge_c/2.0+gce/2.0, edge_c/2.0, gce)
    num_gp = np.size(coor)**3
    gc1, gc2, gc3 = np.meshgrid(coor, coor, coor, indexing='xy')
    gc1, gc2, gc3 = gc1.reshape(1, num_gp)[0], gc2.reshape(1, num_gp)[0], gc3.reshape(1, num_gp)[0]
    grid = np.column_stack([gc1, gc2, gc3])

    return grid, gce

def mis_in_grid(M, Id, grid, gce, tol=1e-06):
    mis = np.copy(M); id = np.copy(Id)
    mig = np.zeros((len(grid), 3))
    idg = np.zeros((len(grid), 3))
    for ct1 in range(len(grid)):
        print ct1
        gp = grid[ct1, :]

        xlu = gp[0] + gce/2.0
        xll = gp[0] - gce/2.0
        ylu = gp[1] + gce/2.0
        yll = gp[1] - gce/2.0
        zlu = gp[2] + gce/2.0
        zll = gp[2] - gce/2.0

        cond_xl = xll - mis[:, 0] < tol
        cond_xu = mis[:, 0] - xlu < tol
        condx = cond_xl & cond_xu

        cond_yl = yll - mis[:, 1] < tol
        cond_yu = mis[:, 1] - ylu < tol
        condy = cond_yl & cond_yu

        cond_zl = zll - mis[:, 2] < tol
        cond_zu = mis[:, 2] - zlu < tol
        condz = cond_zl & cond_zu

        cond = condx & condy & condz
        mis_g = mis[cond]; id_g = id[cond]
        if np.any(cond):
            mis_c, id_c = pick_mis_ctr(mis_g, id_g, gp)

            mig[ct1, :] = mis_c
            idg[ct1, :] = id_c

            del_ind = np.where((mis[:, 0] == mis_c[0]) & (mis[:, 1] == mis_c[1]) & (mis[:, 2] == mis_c[2]))[0][0]
            mis = np.delete(mis, del_ind, 0)
            id = np.delete(id, del_ind, 0)
        else:
            raise "Not enough misorientations to map!"
    return mig, idg

def pick_mis_ctr(mis, id, g, tol=1e-06):

    if len(mis) == 1:
        return mis[0], id[0]
    else:
        cond = np.abs(id[:, 0] - id[0][0]) < tol
        mis_min = mis[cond]
        num_mis_min = len(mis_min)
        d_ctr = np.zeros(num_mis_min)
        for ct1 in range(num_mis_min):
            pt = mis_min[ct1]
            d_ctr[ct1] = np.sqrt((g[0] - pt[0])**2 + (g[1] - pt[1])**2+ (g[2] - pt[2])**2)
        ind = np.argsort(d_ctr)
        mis_pick = mis_min[ind][0]
        id_pick = id[cond][ind][0]
        return mis_pick, id_pick
pick_uni_mis('cF_Id')

