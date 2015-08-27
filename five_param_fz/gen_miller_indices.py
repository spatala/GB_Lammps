import numpy as np
import GBpy.integer_manipulations as GBim
import GBpy.tools as GBt
def gen_miller_ind(mesh_size):
    """
    Returns an array of unique miller indices

    Parameters
    ----------
    mesh_size: size of the mesh grid to create the indices array
    * positive integer

    Returns
    -------
    mil_ind: array of unique miller indices stored row wise
    * numpy array of size (m x 3)
    * m is the number of unique indices created for a given mesh_size

    See Also
    --------
    * GBpy.integer_manipulations.int_finder
    * GBpy.tools.unique_rows_tol
    """

    ### Creating an array of boundary plane miller indices
    r = mesh_size
    x_csl = np.arange(-r, r+1, 1.0)
    y_csl = np.arange(-r, r+1, 1.0)
    z_csl = np.arange(-r, r+1, 1.0)
    num_mi = np.size(x_csl)*np.size(y_csl)*np.size(z_csl)
    xx_csl, yy_csl, zz_csl = np.meshgrid(x_csl, y_csl, z_csl, indexing='xy')
    xx_csl, yy_csl, zz_csl = xx_csl.reshape(1, num_mi)[0], yy_csl.reshape(1, num_mi)[0], zz_csl.reshape(1, num_mi)[0]
    mil_ind = np.column_stack([xx_csl, yy_csl, zz_csl])
    ind = np.where((mil_ind[:, 0] == 0) & (mil_ind[:, 1] == 0) & (mil_ind[:, 2] == 0))[0][0]
    ### deleting (0 0 0)
    mil_ind = np.delete(mil_ind, ind, 0)
    ### finding the unique miller indices
    mil_ind = GBim.int_finder(mil_ind, tol=1e-06, order='rows')
    mil_ind = GBt.unique_rows_tol(mil_ind)
    # Try to remove (-h -k -l) for all (h k l) !!!
    return mil_ind
gen_miller_ind(3)