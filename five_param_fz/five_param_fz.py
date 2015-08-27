import numpy as np
import GBpy.quaternion as GBq
from five_param_symm_props import five_param_symm_props
from pick_fz_bpl import pick_fz_bpl

def five_param_fz(r_go1togo2_go1, bpn_go1):
    """The function returns an array of unit boundary plane normals lying in the fundamental zone of bicrystal
    symmetry for an input array containing boundary plane normals in po1 reference frame.

    Parameters
    -----------
    r_go1togo2_go1: The sigma misorientation matrix for grain boundary in orthogonal coordinates measured in the
    reference frame of crystal 1 (po1).
    * numpy array of size (3 x 3)

    bpn_go1: Boundary plane array containing boundary plane vectors, stored row wise.
    * numpy array of size (n x 3)
    * For each vector the components are expressed in orthogonal reference frame of crystal 1 (po1).

    Returns
    --------
    bp_fz_norms_go1: Fundamental zone boundary plane array.
    * numpy array of size (n x 3)
    * Each row is a unit boundary plane vector in the bicrystal fundamental zone.
    * For each vector the components are expressed in orthogonal reference frame of crystal 1 (po1).

    symm_grp_ax: The principal axes of bicrystal symmetry group in orthogonal reference frame of crystal 1 (po1).
    * numpy array of size (3 x 3)
    * x_axis == symm_grp_axes[:, 0]; y_axis == symm_grp_axes[:, 1]; z_axis == symm_grp_axes[:, 2]

    bp_symm_grp: The bicrystal symmetry group of the grain boundary.
    * python string
    * allowed values limited to 'C_s', 'C_2h', 'D_3d', 'D_2h', 'D_4h', 'D_6h', 'D_8h' and 'O_h'.

    Notes
    -------
    * all inputs and outputs, for this method are in the reference frame of lower crystal 1 (po1).
    * bpn_go1 should always be input as a 2-D array; to input just one normal, say [1, 0, 0], use the
    following syntax, bpn_go1 = np.array([[1, 0, 0]]).

    See Also
    --------
    * GBpy.quaternion
    * five_param_symm_props
    * pick_fz_bpl
    """
    mis_quat_fz = GBq.mat2quat(r_go1togo2_go1); lat_pt_grp = 'O_h'
    x_g, y_g, z_g, bp_symm_grp = five_param_symm_props(mis_quat_fz, lat_pt_grp, 1e-04)
    symm_grp_ax = np.zeros((3, 3))
    symm_grp_ax[:, 0] = x_g; symm_grp_ax[:, 1] = y_g; symm_grp_ax[:, 2] = z_g
    ### normalizing the input boundary plane vectors array
    t1_vecs = bpn_go1
    new_col = np.linalg.norm(t1_vecs, axis=1)
    t1_vecs_norm = np.array([new_col,]*3).T
    t1_vecs = np.divide(t1_vecs, t1_vecs_norm)
    bpn_go1 = t1_vecs

    bp_fz_norms_go1, bp_fz_stereo = pick_fz_bpl(bpn_go1, bp_symm_grp, symm_grp_ax, 1e-04)

    return bp_fz_norms_go1, bp_symm_grp, symm_grp_ax
# r = np.array([[0.6666667, -0.3333333, 0.6666667], [0.6666667, 0.6666667, -0.3333333], [-0.3333333, 0.6666667, 0.6666667]])
# b = np.array([[0, -3, -1]])
# dum = five_param_fz(r, b)
# print dum

# r = np.array([[0.6666667, -0.3333333, 0.6666667], [0.6666667, 0.6666667, -0.3333333], [-0.3333333, 0.6666667, 0.6666667]])
# b = np.array([[3], [1], [0]])
# print five_param_fz(r, b)
