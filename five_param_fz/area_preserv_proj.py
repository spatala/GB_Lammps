import numpy as np
def area_preserv_proj(pts, rot_mat, tol=1e-08):
    """
    Rotates the input points to a new axes system and takes the area preseving projection along the z-axis.

    Parameters
    ----------
    pts: input points stored in an array in a row wise fashion
    * numpy array of size (n x 3)

    rot_mat: the coordinate axes of the new system
    * numpy array of size (3 x3)
    * z-axis is along rot_mat[:, 2], y-axis is along rot_mat[:, 1], x-axis is along rot_mat[:, 0]

    tol: minimum tolerance to distinguish positive and negative z-coordinates
    * float with default value 1e-08

    Returns
    -------
    st_pos: stereographic projection of the points above the xy plane of new coordinate axes
    * numpy array of size (n x 2)

    st_neg: stereographic projection of the points below the xy plane of new coordinate axes
    * numpy array of size (n x 2)
    """
    m1 = np.linalg.inv(rot_mat)
    pts_rot = (np.dot(m1, pts.transpose())). transpose()
    pts_pos = np.copy(pts_rot[pts_rot[:, 2] >= -tol, :])
    pts_neg = np.copy(pts_rot[pts_rot[:, 2] <= tol, :])

    ap_pos = []
    ap_neg = []
    if np.size(pts_pos) > 0:
        pts_phi = np.arctan2(pts_pos[:, 1], pts_pos[:, 0])
        a = np.sqrt(2*(1 - np.abs(pts_pos[:, 2])))*np.cos(pts_phi)
        b = np.sqrt(2*(1 - np.abs(pts_pos[:, 2])))*np.sin(pts_phi)
        ap_pos = np.column_stack((a, b))
    if np.size(pts_neg) > 0:
        pts_phi = np.arctan2(pts_neg[:, 1], pts_neg[:, 0])
        a1 = np.sqrt(2*(1 - np.abs(pts_neg[:, 2])))*np.cos(pts_phi)
        b1 = np.sqrt(2*(1 - np.abs(pts_neg[:, 2])))*np.sin(pts_phi)
        ap_neg = np.column_stack((a1, b1))

    return ap_pos, ap_neg
