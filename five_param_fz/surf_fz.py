import numpy as np
import hkl as hkl
import os
from plot_2d_fz import plot_2d_fz
import pickle
import GBpy.lattice as GBl
# import pick_uni_bpn as pub
#
# from pick_uni_bpn import pick_uni_bpn

def gen_plot(r, sig, g):
    #### Function Call to hkl.py #####
    # pts_m, rot_mat_m, symm_grp_m, s_m = hkl.get_bpn(r, sig, g)
    elem = GBl.Lattice('cF_Id')
    l_g_go = elem.l_p_po
    pkl_path = 'Sig_3_79.pkl'
    pkl_content = pickle.load(open(pkl_path))
    i = 0; j = 0
    pts_m = pkl_content[i][j][2][:, [1, 2, 3]]
    rot_mat_m = pkl_content[i][j][5]
    rot_mat_m = np.dot(l_g_go, rot_mat_m)
    symm_grp_m = pkl_content[i][j][4]
    s_m = 'S'; r = 16; sig = 3
    # pts_m, rot_mat_m, symm_grp_m = pick_uni_bpn(r, sig, 'cF_Id')

    plots = os.getcwd()+'/plots'
    # if os.path.exists(plots):
    #     pass
    # else:
    #     os.mkdir(plots)

    # sigma_path = plots + '/sig_' + str(sig)
    # if os.path.exists(sigma_path):
    #     pass
    # else:
    #     os.mkdir(sigma_path)

    # r_path = sigma_path + '/range_'+ str(r)
    # r_path = r_path.replace(' ', '\\ ')

    # if os.path.exists(r_path):
    #     os.system('rm -R '+r_path)
    #     os.mkdir(r_path)
    # else:
    #     os.mkdir(r_path)

    if s_m == 'S':
        symm_grp = symm_grp_m
        if symm_grp == 'C_s':
            ang = 2*np.pi
        elif symm_grp == 'C_2h':
            ang = np.pi
        elif symm_grp == 'D_3d':
            ang = 'D_3d'
        elif symm_grp == 'D_2h':
            ang = np.pi/2
        elif symm_grp == 'D_4h':
            ang = np.pi/4
        elif symm_grp == 'D_6h':
            ang = np.pi/6
        elif symm_grp == 'D_8h':
            ang = np.pi/8
        elif symm_grp == 'O_h':
            ang = 'O_h'
        fig_path = plot_2d_fz(rot_mat_m, pts_m, ang, 1, sig, r, 0)
        # os.system('mv '+fig_path + ' '+r_path)
    else:
        for ct1 in range(np.shape(rot_mat_m)[0]):
            rot_mat_pf = rot_mat_m[ct1]
            symm_grp = symm_grp_m[ct1]
            pts_pf = pts_m[ct1]
            if symm_grp == 'C_s':
                ang = 2*np.pi
            elif symm_grp == 'C_2h':
                ang = np.pi
            elif symm_grp == 'D_3d':
                ang = np.pi/6
            elif symm_grp == 'D_2h':
                ang = np.pi/2
            elif symm_grp == 'D_4h':
                ang = np.pi/4
            elif symm_grp == 'D_6h':
                ang = np.pi/6
            elif symm_grp == 'D_8h':
                ang = np.pi/8
            elif symm_grp == 'O_h':
                ang = 'O_h'
            fig_path = plot_2d_fz(rot_mat_pf, pts_pf, ang, ct1+1, sig, r, 0)
            # os.system('mv '+fig_path + ' '+r_path)
    return plots

# sig_l = list(np.arange(51, 200, 2))
sig_l = [3]
range_l = [50]
g = 300

for sig_val in sig_l:
    for range_val in range_l:
        print '\nPlotting for Sigma ', sig_val, '\n'
        print 'Range ', range_val, '\n'
        p_path = gen_plot(range_val, sig_val, g)
# print '\n\n Runs completed. Plots saved in "' + p_path +'" directory.'

