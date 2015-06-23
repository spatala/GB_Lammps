import numpy as np
import matplotlib.pyplot as plt
# from hkl import get_bpn as hkl_data
import hkl as hkl
import os

def area_preserv_proj(pts, rot_mat, tol=1e-08):
    """
    Rotate the points such that 
    z-axis is along rot_mat[:, 2]
    y-axis is along rot_mat[:, 1]
    x-axis is along rot_mat[:, 0]

    The stereographic projection is along z-axis
    """
    m1 = np.linalg.inv(rot_mat)
    pts_rot = (np.dot(m1, pts.transpose())). transpose()
    pts_pos = np.copy(pts_rot[pts_rot[:, 2] >= -tol, :])
    pts_neg = np.copy(pts_rot[pts_rot[:, 2] <= tol, :])
    
    st_pos = []
    st_neg = []
    if np.size(pts_pos) > 0:
        pts_phi = np.arctan2(pts_pos[:, 1], pts_pos[:, 0])
        a = np.sqrt(2*(1 - np.abs(pts_pos[:, 2])))*np.cos(pts_phi)
        b = np.sqrt(2*(1 - np.abs(pts_pos[:, 2])))*np.sin(pts_phi)
        st_pos = np.column_stack((a, b))
    if np.size(pts_neg) > 0:
        pts_phi = np.arctan2(pts_neg[:, 1], pts_neg[:, 0])
        a1 = np.sqrt(2*(1 - np.abs(pts_neg[:, 2])))*np.cos(pts_phi)
        b1 = np.sqrt(2*(1 - np.abs(pts_neg[:, 2])))*np.sin(pts_phi)
        st_neg = np.column_stack((a1, b1))

    return st_pos, st_neg

def plot_fig(rot_mat_m, pts_m, ang, ct1, sig, r):

    fig = plt.figure(figsize=(12, 12), facecolor='w')
    plt.margins(0.05)
    rot_mat = rot_mat_m
    pts = pts_m
    tout = area_preserv_proj(pts, rot_mat, tol=1e-08)
    pt_stereo1 = tout[0]
    plt.plot(pt_stereo1[:, 0], pt_stereo1[:, 1], 'o', \
             markerfacecolor='red', markeredgecolor='blue', markersize=5)

    ##############################################################
    ### Boundaries of the surface Fundamental zone
    if ang == 'O_h':
        ### Special case for O_h bicrystal symmetry boundaries
        rot_mat = np.array([[0,0,1], [1,0,0], [0,1,0]])
        num1 = 100
        tphi = np.linspace(0, np.pi/4, num1)
        tpts1 = np.zeros((num1, 3))
        tpts1[:, 0] = np.cos(tphi)
        tpts1[:, 1] = np.sin(tphi)
        tout = area_preserv_proj(tpts1, rot_mat, tol=1e-08)
        tpts1_stereo = tout[0]
        plt.plot(tpts1_stereo[:, 0], tpts1_stereo[:, 1], color=(0, 0, 0))

        num1 = 100
        th1 = np.arctan(np.sqrt(2))
        tphi = np.linspace(np.pi/4, np.pi/4, num1)
        ttheta = np.linspace(th1, np.pi/2, num1)
        tpts1 = np.zeros((num1, 3))
        tpts1[:, 0] = np.cos(tphi)*np.sin(ttheta)
        tpts1[:, 1] = np.sin(tphi)*np.sin(ttheta)
        tpts1[:, 2] = np.cos(ttheta)
        tout = area_preserv_proj(tpts1, rot_mat, tol=1e-08)
        tpts1_stereo = tout[0]
        plt.plot(tpts1_stereo[:, 0], tpts1_stereo[:, 1], color=(0, 0, 0))

        num1 = 100
        th1 = np.arctan(np.sqrt(2))
        tphi = np.linspace(0, np.pi/4, num1)
        ttheta = np.arctan(1/np.sin(tphi))
        tpts1 = np.zeros((num1, 3))
        tpts1[:, 0] = np.cos(tphi)*np.sin(ttheta)
        tpts1[:, 1] = np.sin(tphi)*np.sin(ttheta)
        tpts1[:, 2] = np.cos(ttheta)
        tout = area_preserv_proj(tpts1, rot_mat, tol=1e-08)
        tpts1_stereo = tout[0]
        plt.plot(tpts1_stereo[:, 0], tpts1_stereo[:, 1], color=(0, 0, 0))

    elif ang == 'D_3d':
        rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        num1 = 100
        phi = np.linspace(-np.pi/6, np.pi/6, num1)
        theta = np.linspace(np.pi/2, np.pi/2, num1)
        tpts = np.zeros((num1, 3))
        tpts[:, 0] = np.sin(theta)*np.cos(phi)
        tpts[:, 1] = np.sin(theta)*np.sin(phi)
        tpts[:, 2] = np.cos(theta)
        tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
        tpts_stereo = tout[0]
        plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

        num1 = 100
        phi = np.linspace(-np.pi/6, -np.pi/6, num1)
        theta = np.linspace(0, np.pi/2, num1)
        tpts = np.zeros((num1, 3))
        tpts[:, 0] = np.sin(theta)*np.cos(phi)
        tpts[:, 1] = np.sin(theta)*np.sin(phi)
        tpts[:, 2] = np.cos(theta)
        tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
        tpts_stereo = tout[0]
        plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

        num1 = 100
        phi = np.linspace(np.pi/6, np.pi/6, num1)
        theta = np.linspace(0, np.pi/2, num1)
        tpts = np.zeros((num1, 3))
        tpts[:, 0] = np.sin(theta)*np.cos(phi)
        tpts[:, 1] = np.sin(theta)*np.sin(phi)
        tpts[:, 2] = np.cos(theta)
        tout = area_preserv_proj(tpts, rot_mat, tol=1e-08)
        tpts_stereo = tout[0]
        plt.plot(tpts_stereo[:, 0], tpts_stereo[:, 1], color=(0, 0, 0))

    else:
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

        if abs(ang - 2*np.pi) > 1e-08:
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

    ##############################################################

    plt.axis('equal')
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig('Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'.png', dpi=100, bbox_inches='tight')
    fig_path = os.getcwd() + '/Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'.png'
    return fig_path

def gen_plot(r, sig):
    #### Function Call to hkl.py #####
    pts_m, rot_mat_m, symm_grp_m, s_m = hkl.get_bpn(r, sig)

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
        fig_path = plot_fig(rot_mat_m, pts_m, ang, 1, sig, r)
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
            fig_path = plot_fig(rot_mat_pf, pts_pf, ang, ct1+1, sig, r)
            # os.system('mv '+fig_path + ' '+r_path)
    return plots

# sig_l = list(np.arange(51, 200, 2))
sig_l = [3]
range_l = [1]
for sig_val in sig_l:
    for range_val in range_l:
        print '\nPlotting for Sigma ', sig_val, '\n'
        print 'Mesh Grid Range ', range_val, '\n'
        p_path = gen_plot(range_val, sig_val)
print '\n\n Runs completed. Plots saved in "' + p_path +'" directory.'

