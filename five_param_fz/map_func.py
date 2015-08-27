import numpy as np
import math
import GBpy.quaternion as GBqt
import GBpy.tools as GBt
def sphr2cube_2d(pt):
    pt_c_2d = ball2cube(pt)
    pt_c = m1(pt_c_2d)
    return pt_c

def cube2sphr_2d(pt):
    pt_c_2d = m1_inv(pt)
    pt_s = cube2ball(pt_c_2d)
    return pt_s

def sphr2cube_3d(quat):
    t = GBqt.getq0(quat); x = GBqt.getq1(quat); y = GBqt.getq2(quat); z = GBqt.getq3(quat)
    f_of_t = (1.0/np.sqrt(1-t*t))*((3.0/2)*(np.arccos(t) - t*np.sqrt(1- t*t)))**(1.0/3)
    xx = f_of_t*x; yy = f_of_t*y; zz = f_of_t*z
    pt_ball = np.column_stack((xx, yy, zz))
    pt_cube = ball2cube(pt_ball)

    return pt_cube

def sphr2cube_3dv2(quat):
    theta = 2*np.arccos(GBqt.getq0(quat))
    f_of_t = (0.75*(theta - np.sin(theta)))**(1.0/3)
    xyz = np.column_stack([GBqt.getq1(quat), GBqt.getq2(quat), GBqt.getq3(quat)])/np.sin(theta/2.0).reshape(len(theta), 1)
    XYZ = f_of_t.reshape(len(f_of_t), 1)*xyz
    pt_cube = ball2cube(XYZ)
    return pt_cube
def cube2sphr_3d(pt):
    pt_ball = cube2ball(pt)
    rho_sq = pt_ball[:, 0]**2 + pt_ball[:, 1]**2 + pt_ball[:, 2]**2
    t = 1 + (-0.500009615)*(rho_sq) + (-0.024866061)*(rho_sq**2) + (-0.004549382)*(rho_sq**3) +(0.000511867)*(rho_sq**4)\
               + (-0.001650083)*(rho_sq**5) + (0.000759335)*(rho_sq**6) + (-0.000204042)*(rho_sq**7)
    q123 = pt_ball*(np.sqrt((1-t*t)/rho_sq)).reshape(len(t), 1)
    inp_quat = np.hstack([t.reshape(len(t), 1), q123])
    quat_sphr = GBqt.Quaternion(inp_quat)
    # ax_ang = np.hstack((axis, 2*np.arccos(t).reshape(len(t), 1)))
    # quat_sphr = GBt.axang2quat(ax_ang)

    return quat_sphr

def ball2cube(pt):

    pts = np.copy(pt)

    cond1a = pts[:, 2] >= 0
    cond1b = pts[:, 2] >= np.abs(pts[:, 0])
    cond1c = pts[:, 2] >= np.abs(pts[:, 1])
    cond1 = cond1a & cond1b & cond1c
    pts1 = pts[cond1]; pts = np.delete(pts, np.where(cond1)[0], 0)

    cond2a = pts[:, 0] > 0
    cond2b = pts[:, 0] > np.abs(pts[:, 2])
    cond2c = pts[:, 0] >= np.abs(pts[:, 1])
    cond2 = cond2a & cond2b & cond2c
    pts2 = pts[cond2]; pts = np.delete(pts, np.where(cond2)[0], 0)

    cond3a = pts[:, 1] > 0
    cond3b = pts[:, 1] > np.abs(pts[:, 2])
    cond3c = pts[:, 1] > np.abs(pts[:, 0])
    cond3 = cond3a & cond3b & cond3c
    pts3 = pts[cond3]; pts = np.delete(pts, np.where(cond3)[0], 0)

    cond4a = pts[:, 2] < 0
    cond4b = -pts[:, 2] >= np.abs(pts[:, 0])
    cond4c = -pts[:, 2] >= np.abs(pts[:, 1])
    cond4 = cond4a & cond4b & cond4c
    pts4 = pts[cond4]; pts = np.delete(pts, np.where(cond4)[0], 0)

    cond5a = pts[:, 0] < 0
    cond5b = -pts[:, 0] > np.abs(pts[:, 2])
    cond5c = -pts[:, 0] >= np.abs(pts[:, 1])
    cond5 = cond5a & cond5b & cond5c
    pts5 = pts[cond5]; pts = np.delete(pts, np.where(cond5)[0], 0)

    cond6a = pts[:, 1] < 0
    cond6b = -pts[:, 1] > np.abs(pts[:, 2])
    cond6c = -pts[:, 1] > np.abs(pts[:, 0])
    cond6 = cond6a & cond6b & cond6c
    pts6 = pts[cond6]; pts = np.delete(pts, np.where(cond6)[0], 0)

    if len(pts) != 0:
        raise "Algorithm is incorrect!"

    pts_cube = np.zeros(np.shape(pt))

    if len(pts1) > 0:
        pts1_m3 = m3_inv(pts1, (0, 0, 1), 1)
        pts1_m2 = m2_inv(pts1_m3, (0, 0, 1), 1)
        pts1_m1 = m1_inv(pts1_m2)
        ind1 = set_int_ind(pt, pts1)
        pts_cube[ind1, :] = pts1_m1

    if len(pts2) > 0:
        pts2_m3 = m3_inv(pts2, (1, 0, 0), 1)
        pts2_m2 = m2_inv(pts2_m3, (1, 0, 0), 1)
        pts2_m1 = m1_inv(pts2_m2)
        ind2 = set_int_ind(pt, pts2)
        pts_cube[ind2, :] = pts2_m1

    if len(pts3) > 0:
        pts3_m3 = m3_inv(pts3, (0, 1, 0), 1)
        pts3_m2 = m2_inv(pts3_m3, (0, 1, 0), 1)
        pts3_m1 = m1_inv(pts3_m2)
        ind3 = set_int_ind(pt, pts3)
        pts_cube[ind3, :] = pts3_m1

    if len(pts4) > 0:
        pts4_m3 = m3_inv(pts4, (0, 0, 1), -1)
        pts4_m2 = m2_inv(pts4_m3, (0, 0, 1), -1)
        pts4_m1 = m1_inv(pts4_m2)
        ind4 = set_int_ind(pt, pts4)
        pts_cube[ind4, :] = pts4_m1

    if len(pts5) > 0:
        pts5_m3 = m3_inv(pts5, (1, 0, 0), -1)
        pts5_m2 = m2_inv(pts5_m3, (1, 0, 0), -1)
        pts5_m1 = m1_inv(pts5_m2)
        ind5 = set_int_ind(pt, pts5)
        pts_cube[ind5, :] = pts5_m1

    if len(pts6) > 0:
        pts6_m3 = m3_inv(pts6, (0, 1, 0), -1)
        pts6_m2 = m2_inv(pts6_m3, (0, 1, 0), -1)
        pts6_m1 = m1_inv(pts6_m2)
        ind6 = set_int_ind(pt, pts6)
        pts_cube[ind6, :] = pts6_m1


    return pts_cube

def cube2ball(pt):

    pts = np.copy(pt)
    cond1a = pts[:, 2] >= 0
    cond1b = pts[:, 2] >= np.abs(pts[:, 0])
    cond1c = pts[:, 2] >= np.abs(pts[:, 1])
    cond1 = cond1a & cond1b & cond1c
    pts1 = pts[cond1]; pts = np.delete(pts, np.where(cond1)[0], 0)

    cond2a = pts[:, 0] > 0
    cond2b = pts[:, 0] > np.abs(pts[:, 2])
    cond2c = pts[:, 0] >= np.abs(pts[:, 1])
    cond2 = cond2a & cond2b & cond2c
    pts2 = pts[cond2]; pts = np.delete(pts, np.where(cond2)[0], 0)

    cond3a = pts[:, 1] > 0
    cond3b = pts[:, 1] > np.abs(pts[:, 2])
    cond3c = pts[:, 1] > np.abs(pts[:, 0])
    cond3 = cond3a & cond3b & cond3c
    pts3 = pts[cond3]; pts = np.delete(pts, np.where(cond3)[0], 0)

    cond4a = pts[:, 2] < 0
    cond4b = -pts[:, 2] >= np.abs(pts[:, 0])
    cond4c = -pts[:, 2] >= np.abs(pts[:, 1])
    cond4 = cond4a & cond4b & cond4c
    pts4 = pts[cond4]; pts = np.delete(pts, np.where(cond4)[0], 0)

    cond5a = pts[:, 0] < 0
    cond5b = -pts[:, 0] > np.abs(pts[:, 2])
    cond5c = -pts[:, 0] >= np.abs(pts[:, 1])
    cond5 = cond5a & cond5b & cond5c
    pts5 = pts[cond5]; pts = np.delete(pts, np.where(cond5)[0], 0)

    cond6a = pts[:, 1] < 0
    cond6b = -pts[:, 1] > np.abs(pts[:, 2])
    cond6c = -pts[:, 1] > np.abs(pts[:, 0])
    cond6 = cond6a & cond6b & cond6c
    pts6 = pts[cond6]; pts = np.delete(pts, np.where(cond6)[0], 0)


    if len(pts) != 0:
        raise " Algorithm is incorrect!"

    pts_ball = np.zeros(np.shape(pt))
    if len(pts1) > 0:
        pts1_m1 = m1(pts1)
        pts1_m2 = m2(pts1_m1, (0, 0, 1), 1)
        pts1_m3 = m3(pts1_m2, (0, 0, 1), 1)
        ind1 = set_int_ind(pt, pts1)
        pts_ball[ind1, :] = pts1_m3

    if len(pts2) > 0:
        pts2_m1 = m1(pts2)
        pts2_m2 = m2(pts2_m1, (1, 0, 0), 1)
        pts2_m3 = m3(pts2_m2, (1, 0, 0), 1)
        ind2 = set_int_ind(pt, pts2)
        pts_ball[ind2, :] = pts2_m3

    if len(pts3) > 0:
        pts3_m1 = m1(pts3)
        pts3_m2 = m2(pts3_m1, (0, 1, 0), 1)
        pts3_m3 = m3(pts3_m2, (0, 1, 0), 1)
        ind3 = set_int_ind(pt, pts3)
        pts_ball[ind3, :] = pts3_m3

    if len(pts4) > 0:
        pts4_m1 = m1(pts4)
        pts4_m2 = m2(pts4_m1, (0, 0, 1), -1)
        pts4_m3 = m3(pts4_m2, (0, 0, 1), -1)
        ind4 = set_int_ind(pt, pts4)
        pts_ball[ind4, :] = pts4_m3

    if len(pts5) > 0:
        pts5_m1 = m1(pts5)
        pts5_m2 = m2(pts5_m1, (1, 0, 0), -1)
        pts5_m3 = m3(pts5_m2, (1, 0, 0), -1)
        ind5 = set_int_ind(pt, pts5)
        pts_ball[ind5, :] = pts5_m3

    if len(pts6) > 0:
        pts6_m1 = m1(pts6)
        pts6_m2 = m2(pts6_m1, (0, 1, 0), -1)
        pts6_m3 = m3(pts6_m2, (0, 1, 0), -1)
        ind6 = set_int_ind(pt, pts6)
        pts_ball[ind6, :] = pts6_m3

    return pts_ball

def m1(pts):

    coeff = (np.pi/6)**(1.0/6)
    pts_r = coeff*pts

    return pts_r

def m2(pts, plane, pl_type):
    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
        z = pts[:, 1]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 0]

    cond1 = np.abs(y) <= np.abs(x)
    x1 = x[cond1]
    y1 = y[cond1]
    # zt_1 = z[cond1]
    ind1 = np.where(cond1)[0]

    cond2 = np.abs(x) < np.abs(y)
    x2 = x[cond2]
    y2 = y[cond2]
    # zt_2 = z[cond2]
    ind2 = np.where(cond2)[0]

    beta = math.sqrt(np.pi/6)

    if np.shape(x1)[0] > 0:
        ind_x0 = np.where(x1 == 0)[0]
        xt_1 = ((2**0.25)/beta)*x1*(math.sqrt(2)*np.cos((np.pi*y1/(12*x1)))-1)/np.sqrt(math.sqrt(2)-np.cos((np.pi*y1/(12*x1))))
        yt_1 = ((2**0.75)/beta)*x1*np.sin((np.pi*y1/(12*x1)))/np.sqrt(math.sqrt(2)-np.cos((np.pi*y1/(12*x1))))
        if len(ind_x0)>0:
            xt_1[ind_x0] = 0
            yt_1[ind_x0] = y1[ind_x0]

    if np.shape(x2)[0] > 0:
        ind_y0 = np.where(y2 == 0)[0]
        yt_2 = ((2**0.25)/beta)*y2*(math.sqrt(2)*np.cos((np.pi*x2/(12*y2)))-1)/np.sqrt(math.sqrt(2)-np.cos((np.pi*x2/(12*y2))))
        xt_2 = ((2**0.75)/beta)*y2*np.sin((np.pi*x2/(12*y2)))/np.sqrt(math.sqrt(2)-np.cos((np.pi*x2/(12*y2))))
        if len(ind_y0) > 0:
            xt_2[ind_y0] = x2[ind_y0]
            yt_2[ind_y0] = 0

    if np.shape(x1)[0] == 0:
        xt = xt_2
        yt = yt_2
        # zt = zt_2
    elif np.shape(x2)[0] == 0:
        xt = xt_1
        yt = yt_1
        # zt = zt_1
    else:
        xt = np.zeros(np.shape(x))
        yt = np.zeros(np.shape(y))
        # zt = np.zeros(np.shape(z))

        xt[ind1] = xt_1
        xt[ind2] = xt_2
        yt[ind1] = yt_1
        yt[ind2] = yt_2
        # zt[ind1] = zt_1
        # zt[ind2] = zt_2

    zt = z

    if plane == (0, 0, 1):
        pts_r = np.column_stack((xt, yt, zt))
    elif plane == (0, 1, 0):
        pts_r = np.column_stack((yt, zt, xt))
    elif plane == (1, 0, 0):
        pts_r = np.column_stack((zt, xt, yt))

    return pts_r

def m3(pts, plane, pl_type):
    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
        z = pts[:, 1]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 0]

    xl = ((1.0 - (x*x + y*y)*(np.pi/(24.0*z*z)))**0.5) * x
    yl = ((1.0 - (x*x + y*y)*(np.pi/(24.0*z*z)))**0.5) * y
    zl = ((6.0/np.pi)**0.5)*z - ((x*x + y*y)/z)*((np.pi/24.0)**0.5)

    ###Taking care of origin
    ind_000 = np.where(z==0)[0]
    if len(ind_000 !=0):
        xl[ind_000], yl[ind_000], zl[ind_000] = 0, 0, 0
    # if pl_type == -1:
    #     zl = -zl

    if plane == (0, 0, 1):
        pts_r = np.column_stack((xl, yl, zl))
    elif plane == (0, 1, 0):
        pts_r = np.column_stack((yl, zl, xl))
    elif plane == (1, 0, 0):
        pts_r = np.column_stack((zl, xl, yl))
    return pts_r

def m1_inv(pts):
    coeff = (np.pi/6)**(1.0/6)
    pts_r = pts/coeff
    return pts_r

def m2_inv(pts, plane, pl_type):
    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
        z = pts[:, 1]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 0]

    cond1 = np.abs(y) <= np.abs(x)
    x1 = x[cond1]
    y1 = y[cond1]
    # zt_1 = z[cond1]
    ind1 = np.where(cond1)[0]

    cond2 = np.abs(x) < np.abs(y)
    x2 = x[cond2]
    y2 = y[cond2]
    # zt_2 = z[cond2]
    ind2 = np.where(cond2)[0]

    beta = math.sqrt(np.pi/6)

    if np.shape(x1)[0] > 0:

        tx1 = beta/math.sqrt(2)
        tx2 = np.sign(x1)
        tx3 = (2*x1*x1 + y1*y1)**0.25
        tx4 = (np.abs(x1)+(2*x1*x1 + y1*y1)**0.5)**0.5
        xt_1 = tx1*tx2*tx3*tx4

        ty1 = 1/tx1
        ty2 = tx3
        ty3 = tx4
        ind_x0 = np.where(x1 == 0)[0]
        ind_x0_y0 = list(set(ind_x0).intersection(np.where(y1 == 0)[0]))
        ty4 = (np.sign(x1)*np.arctan(y1/x1) - np.arctan(y1/((2*x1*x1+y1*y1)**0.5)))

        if len(ind_x0) > 0:
            ty4[ind_x0] = - np.arctan(y1[ind_x0]/np.abs(y1[ind_x0]))
        if len(ind_x0_y0) > 0:
            ty4[ind_x0_y0] = 0

        yt_1 = ty1*ty2*ty3*ty4

    if np.shape(x2)[0] > 0:
        tx1 = math.sqrt(2)/beta
        tx2 = (x2*x2 + 2*y2*y2)**0.25
        tx3 = (np.abs(y2) + (x2*x2 + 2*y2*y2)**0.5)**0.5

        ind_y0 = np.where(y2 == 0)[0]
        ind_y0_x0 = list(set(ind_y0).intersection(np.where(x2 == 0)[0]))

        tx4 = (np.sign(y2)*np.arctan(x2/y2) - np.arctan(x2/((x2*x2 + 2*y2*y2)**0.5)))

        if len(ind_y0) > 0:
            ty4[ind_y0] = -np.arctan(x2[ind_y0]/np.abs(x2[ind_y0]))
        if len(ind_y0_x0) > 0:
            ty4[ind_y0_x0] = 0

        xt_2 = tx1*tx2*tx3*tx4

        ty1 = 1/tx1
        ty2 = np.sign(y2)
        ty3 = tx2
        ty4 = tx3
        yt_2 = ty1*ty2*ty3*ty4

    if np.shape(x1)[0] == 0:
        xt_inv = xt_2
        yt_inv = yt_2
        # zt_inv = zt_2
    elif np.shape(x2)[0] == 0:
        xt_inv = xt_1
        yt_inv = yt_1
        # zt_inv = zt_1
    else:
        xt_inv = np.zeros(np.shape(x))
        yt_inv = np.zeros(np.shape(y))
        # zt_inv = np.zeros(np.shape(z))

        xt_inv[ind1] = xt_1
        xt_inv[ind2] = xt_2
        yt_inv[ind1] = yt_1
        yt_inv[ind2] = yt_2
        # zt_inv[ind1] = zt_1
        # zt_inv[ind2] = zt_2


    # if pl_type == 1:
    #     zt_inv = z
    # if pl_type == -1:
    #     zt_inv = -z

    zt_inv = z

    if plane == (0, 0, 1):
        pts_r = np.column_stack((xt_inv, yt_inv, zt_inv))
    elif plane == (0, 1, 0):
        pts_r = np.column_stack((yt_inv, zt_inv, xt_inv))
    elif plane == (1, 0, 0):
        pts_r = np.column_stack((zt_inv, xt_inv, yt_inv))

    return pts_r

def m3_inv(pts, plane, pl_type):

    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
        z = pts[:, 1]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]
        z = pts[:, 0]

    # if pl_type == 1:
    #     xl = np.multiply(np.sqrt(np.true_divide(2, (1+z))), x)
    #     yl = np.multiply(np.sqrt(np.true_divide(2, (1+z))), y)
    #     zl = np.zeros(np.shape(z)[0]); zl.fill(1.0)
    # if pl_type == -1:
    #     xl = np.multiply(np.sqrt(np.true_divide(2, (1-z))), x)
    #     yl = np.multiply(np.sqrt(np.true_divide(2, (1-z))), y)
    #     zl = np.zeros(np.shape(z)[0]); zl.fill(-1.0)
    rl = (x*x + y*y + z*z)**0.5
    if pl_type == 1:
        xl = x*((2.0*rl/(rl + z))**0.5)
        yl = y*((2.0*rl/(rl + z))**0.5)
        zl = ((np.pi/6)**0.5)*rl
    if pl_type == -1:
        xl = x*((2.0*rl/(rl - z))**0.5)
        yl = y*((2.0*rl/(rl - z))**0.5)
        zl = -((np.pi/6)**0.5)*rl
    # ### Finding if any of the point is origin
    ind_000 = np.where(rl ==0)[0]
    if len(ind_000) != 0:
        xl[ind_000], yl[ind_000], zl[ind_000] = 0, 0, 0

    if plane == (0, 0, 1):
        pts_r = np.column_stack((xl, yl, zl))
    elif plane == (0, 1, 0):
        pts_r = np.column_stack((yl, zl, xl))
    elif plane == (1, 0, 0):
        pts_r = np.column_stack((zl, xl, yl))

    return pts_r


def set_int_ind(set1, set2):

    set1_v = np.ascontiguousarray(set1).view([('', set1.dtype)] * set1.shape[1]).ravel()
    set2_v = np.ascontiguousarray(set2).view([('', set2.dtype)] * set2.shape[1]).ravel()

    # set1_v = np.ascontiguousarray(set1).view(np.dtype((np.void, set1.dtype.itemsize * set1.shape[1])))
    # set2_v = np.ascontiguousarray(set2).view(np.dtype((np.void, set2.dtype.itemsize * set2.shape[1])))
    ind1 = np.argsort(set1_v)
    sorted_set1_v = set1_v[ind1]
    sorted_index = np.searchsorted(sorted_set1_v, set2_v)

    ind2 = np.take(ind1, sorted_index, mode="clip")
    mask = set1_v[ind2] != set2_v
    ind3 = np.ma.array(ind2, mask=mask)
    return ind3

def check2(r):
    sp_pts = np.zeros((r, 3))

    for ct1 in range(r):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        deno = (x*x + y*y + z*z)**0.5
        xs = x/deno; ys = y/deno; zs = z/deno
        sp_pts[ct1, :] = np.array([xs, ys, zs])

    sp1 = np.column_stack((-sp_pts[:, 0], sp_pts[:, 1], sp_pts[:, 2]))
    sp2 = np.column_stack((sp_pts[:, 0], -sp_pts[:, 1], sp_pts[:, 2]))
    sp3 = np.column_stack((sp_pts[:, 0], sp_pts[:, 1], -sp_pts[:, 2]))
    sp4 = np.column_stack((-sp_pts[:, 0], -sp_pts[:, 1], sp_pts[:, 2]))
    sp5 = np.column_stack((sp_pts[:, 0], -sp_pts[:, 1], -sp_pts[:, 2]))
    sp6 = np.column_stack((-sp_pts[:, 0], sp_pts[:, 1], -sp_pts[:, 2]))
    sp = np.concatenate([sp1, sp2, sp3, sp4, sp5, sp6])
    # max_el =[]
    # for i in range(5):
    #     sp = sp - sp/5
    #     pts_cube = ball2cube(sp)
    #     cp = np.copy(pts_cube)
    #     pts_ball = cube2ball(cp)
    #     print sp, '\n\n', pts_cube, '\n\n', pts_ball
    #     print '\n\n', sp - pts_ball, '\n\n', np.max(sp-pts_ball)
    #     max_el.append(np.max(sp-pts_ball))
    # print '\n', max_el, '\n'
    # max_er = np.array(max_el)
    # print np.max(max_er)
    pt_cube = sphr2cube_2d(sp)
    print pt_cube, '\n'
    pt_sphr = cube2sphr_2d(pt_cube)
    print pt_sphr, '\n', np.max(sp-pt_sphr)
    return True
# check2(100)
# pt_elem = 1.0/(3)**0.5
# pt = np.array([[pt_elem, pt_elem, pt_elem], [-pt_elem, -pt_elem, -pt_elem], [-pt_elem, -pt_elem, pt_elem], [pt_elem, -pt_elem, -pt_elem], [-pt_elem, pt_elem, -pt_elem], [pt_elem, pt_elem, -pt_elem]])
# pt = pt/2.0
# pt = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, 0], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1]])/np.sqrt(3)
#
# print pt, '\n'
# # ptc = ball2cube(pt)
# ptc = sphr2cube_2d(pt)
# print ptc, '\n'
# ptr = cube2sphere_2d(ptc)
# print ptr