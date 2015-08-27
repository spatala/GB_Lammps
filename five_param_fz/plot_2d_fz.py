import os
import numpy as np
import matplotlib.pyplot as plt
from area_preserv_proj import area_preserv_proj

def plot_2d_fz(rot_mat_m, pts_m, ang, ct1, sig, r, n):
    """
    Creates the boundaries for 2-D FZ of the particular bicrystal symmetry and plots the normals in FZ.

    Parameters
    ----------
    rot_mat_m:
    pts_m:
    ang:
    ct1:
    sig:
    r:
    n:

    """

    ####################################################################################################################
    ### Creating the plot figure using matplotlib
    fig = plt.figure(figsize=(12, 12), facecolor='w')
    plt.margins(0.05)
    ### Plotting the bpn using area preserving projection
    rot_mat = rot_mat_m
    pts = pts_m
    tout = area_preserv_proj(pts, rot_mat, tol=1e-08)
    pt_stereo1 = tout[0]
    plt.plot(pt_stereo1[:, 0], pt_stereo1[:, 1], 'o', markerfacecolor='red', markersize=4)

    ####################################################################################################################
    ### Plotting the boundaries of 2d FZ of the bicrystal point group
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
        ### Special case for D_3d bicrystal symmetry boundaries
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
            ### Special case when the FZ is not a sector but a circle
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

    ####################################################################################################################
    ### removing axis
    plt.axis('equal')
    plt.axis('off')
    ### saving the plots
    if n != 0:
        plt.savefig('Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'_N_'+str(n)+'.png', dpi=100, bbox_inches='tight')
        fig_path = os.getcwd() + '/Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'_N_'+str(n)+'.png'
    else:
        plt.savefig('Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'.png', dpi=100, bbox_inches='tight')
        fig_path = os.getcwd() + '/Sig_'+str(sig)+'_Mis_'+str(ct1)+'_R_'+str(r)+'.png'
    return fig_path
