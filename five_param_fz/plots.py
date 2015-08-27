import numpy as np
import os, inspect, pickle
import matplotlib.pyplot as plt
import GBpy
from GBpy import lattice as GBl
from GBpy import find_csl_dsc as GBfcd
from GBpy import bp_basis as GBb2
# from GBpy import integer_manipulations as GBim
from GBpy import tools as GBt
from gen_miller_indices import gen_miller_ind
from five_param_fz import five_param_fz
from area_preserv_proj import area_preserv_proj

# from surf_fz import area_preserv_proj
# from surf_fz import plot_fig

def plots(mesh_size, sigma_val, n, lat_type):

    ####################################################################################################################
    ### Creating an array of boundary plane miller indices in CSL lattice
    mil_ind_csl = gen_miller_ind(mesh_size)

    ####################################################################################################################
    ### Creating an instance of lattice class
    elem = GBl.Lattice(lat_type)
    ### Getting the primitive lattice in orthogonal frame
    l_g_go = elem.l_g_go
    ### Extracting the sigma misorientation from the pickle file
    ### Misorientation is in the primitive frame of associated lattice
    gb_dir = os.path.dirname(inspect.getfile(GBpy))
    pkl_path =  gb_dir + '/pkl_files/cF_Id_csl_common_rotations.pkl'
    pkl_content = pickle.load(open(pkl_path))
    sig_mis_N = pkl_content[str(sigma_val)]['N'][0]
    sig_mis_D = pkl_content[str(sigma_val)]['D'][0]
    sig_mis_g = sig_mis_N/sig_mis_D
    ### Converting the misorientation to orthogonal frame/superlattice of the crystal
    ### Done using similarity transformation
    sig_mis_go = np.dot(np.dot(l_g_go, sig_mis_g), np.linalg.inv(l_g_go)).reshape(1,3,3)[0]
    ### Getting the csl basis in primitive frame
    l_csl_g, l_dsc_g = GBfcd.find_csl_dsc(l_g_go, sig_mis_g)
    ### Converting the csl basis to orthogonal frame
    l_csl_go = np.dot(l_g_go, l_csl_g)
    ### reciprocal csl basis in po frame
    l_rcsl_go = GBfcd.reciprocal_mat(l_csl_go)
    ### Converting the miller indices to normals in po frame
    bpn_go = np.dot(l_rcsl_go, mil_ind_csl.transpose()).transpose()

    ####################################################################################################################
    ### Finding the boundary plane normals in the FZ using five_param_fz
    bp_fz_norms_go1, bp_symm_grp, symm_grp_ax = five_param_fz(sig_mis_go, bpn_go)
    ### Finding unique normals
    bp_fz_norms_go1_unq, bfz_unq_ind = GBt.unique_rows_tol(bp_fz_norms_go1, return_index=True)
    ### Finding the input hkl indices corresponding to unique FZ normals
    mil_ind_csl_unq = mil_ind_csl[bfz_unq_ind]

    ####################################################################################################################
    ### Calculating interplanar distance (d sigma hkl) for unique FZ bpn
    l_rcsl_go = GBfcd.reciprocal_mat(l_csl_go)
    mt_cslr_go = np.dot(l_rcsl_go.transpose(), l_rcsl_go)
    d_inv_sqr = np.diag(np.dot(np.dot(mil_ind_csl_unq, mt_cslr_go),mil_ind_csl_unq.transpose()))
    d_inv = np.sqrt(d_inv_sqr)
    d_sig_hkl = np.true_divide(1, d_inv)

    ####################################################################################################################
    ### Calculating unit cell area for 2-D csl unit cells for unique FZ bpn
    pl_den = []
    num_bpn_unq = np.shape(bp_fz_norms_go1_unq)[0]
    for ct1 in range(num_bpn_unq):
        _, _, pl_den_csl = GBb2.bicryst_planar_den(bp_fz_norms_go1_unq[ct1, :], sig_mis_g, l_g_go, 'normal_go', 'g1')
        pl_den.append(pl_den_csl)
    pl_den = np.array(pl_den)
    a_sig_hkl = np.true_divide(1, pl_den)

    ####################################################################################################################
    ### Checking the csl primitive unit cell volume equality
    v_sig_hkl = np.multiply(d_sig_hkl, a_sig_hkl)
    # print v_sig_hkl
    v_basis = abs(np.linalg.det(l_csl_go))
    if np.all(abs(v_sig_hkl-v_basis) < 1e-04):
        print "The two volumes match!"
    else:
        print " Mismatch!"

    ####################################################################################################################
    ### Sorting attributes in increasing order of 2d csl primitive unit cell area
    ind_area_sort = np.argsort(a_sig_hkl)
    a_sig_hkl_sort = np.sort(a_sig_hkl)
    d_sig_hkl_sort = d_sig_hkl[ind_area_sort]
    bp_fz_norms_go1_unq_sort = bp_fz_norms_go1_unq[ind_area_sort]

    ####################################################################################################################
    ### Check to ensure required number of unique bpn are returned
    if np.shape(bp_fz_norms_go1_unq_sort)[0] < n:
        print "Please input a larger mesh grid or reduce the number of boundaries!"
        n = np.shape(bp_fz_norms_go1_unq_sort)[0]

    ####################################################################################################################
    ### Selecting the lowest 'n' area boundaries and their attributes for plotting
    a_plot = a_sig_hkl_sort[:n]
    pd_plot = np.true_divide(1, a_plot)
    d_plot = d_sig_hkl_sort[:n]
    bp_fz_plot = bp_fz_norms_go1_unq_sort[:n]

    ####################################################################################################################
    ### d vs pd plot
    fig1 = plt.figure(figsize=(12, 12), facecolor='w')
    plt.margins(0.05)
    plt.xlabel('Interplanar spacing')
    plt.ylabel('Planar density of 2D-CSL')
    plt.plot(d_plot, pd_plot, 'ro')
    # plt.show()
    plt.savefig('d_vs_pd_' + str(mesh_size) + '_' + str(n)+ '.png', dpi=100, bbox_inches='tight')

    ####################################################################################################################
    ### FZ plot for the sorted and selected boundaries
    na = '_'+ str(mesh_size) + '_'+ str(n)
    plot_fig(symm_grp_ax, bp_fz_plot, np.pi/6, na)
    # plt.show()
    return


def plot_fig(rot_mat_m, pts_m, ang, na):

    fig = plt.figure(figsize=(12, 12), facecolor='w')
    plt.margins(0.05)
    rot_mat = rot_mat_m
    pts = pts_m
    tout = area_preserv_proj(pts, rot_mat, tol=1e-08)
    pt_stereo1 = tout[0]
    plt.plot(pt_stereo1[:, 0], pt_stereo1[:, 1], 'o', \
             markerfacecolor='red', markersize=2)
    ### General boundary conditions for rest of the bicrystal symmetries
    rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    num1 = 100
    phi = np.linspace(0, ang, num1)
    theta = np.linspace(np.pi/2, np.pi/2, num1)
    tpts = np.zeros((num1, 3))
    tpts[:, 0] = np.sin(theta)*np.cos(phi)
    tpts[:, 1] = np.sin(theta)*np.sin(phi)
    tpts[:, 2] = np.cos(theta)
    tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
    tpts_stereo = tout[0]
    plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

    num1 = 100
    phi = np.linspace(0, 0, num1)
    theta = np.linspace(0, np.pi/2, num1)
    tpts = np.zeros((num1, 3))
    tpts[:, 0] = np.sin(theta)*np.cos(phi)
    tpts[:, 1] = np.sin(theta)*np.sin(phi)
    tpts[:, 2] = np.cos(theta)
    tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
    tpts_stereo = tout[0]
    plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

    num1 = 100
    phi = np.linspace(ang, ang, num1)
    theta = np.linspace(0, np.pi/2, num1)
    tpts = np.zeros((num1, 3))
    tpts[:, 0] = np.sin(theta)*np.cos(phi)
    tpts[:, 1] = np.sin(theta)*np.sin(phi)
    tpts[:, 2] = np.cos(theta)
    tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
    tpts_stereo = tout[0]
    plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

    plt.axis('equal')
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig('min_Sig3_' +'bpn' + na + '.png', dpi=100, bbox_inches='tight')
    return

mesh = 3 #1
sig = 3
n = 7
# bpn_num_check(mesh, sig, n, 'cF_Id')
plots(mesh, sig, n, 'cF_Id')
