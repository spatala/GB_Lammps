import numpy as np
import math

def sphr2cube(pt):

    pts = np.copy(pt)

    cond1a = pts[:, 2] > 0
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
        pts1_l = lamb_area(pts1, (0, 0, 1), 1)
        pts1_t_inv = t_inv(pts1_l, (0, 0, 1), 1)
        ind1 = set_int_ind(pt, pts1)
        pts_cube[ind1, :] = pts1_t_inv

    if len(pts2) > 0:
        pts2_l = lamb_area(pts2, (1, 0, 0), 1)
        pts2_t_inv = t_inv(pts2_l, (1, 0, 0), 1)
        ind2 = set_int_ind(pt, pts2)
        pts_cube[ind2, :] = pts2_t_inv

    if len(pts3) > 0:
        pts3_l = lamb_area(pts3, (0, 1, 0), 1)
        pts3_t_inv = t_inv(pts3_l, (0, 1, 0), 1)
        ind3 = set_int_ind(pt, pts3)
        pts_cube[ind3, :] = pts3_t_inv

    if len(pts4) > 0:
        pts4_l = lamb_area(pts4, (0, 0, 1), -1)
        pts4_t_inv = t_inv(pts4_l, (0, 0, 1), -1)
        ind4 = set_int_ind(pt, pts4)
        pts_cube[ind4, :] = pts4_t_inv

    if len(pts5) > 0:
        pts5_l = lamb_area(pts5, (1, 0, 0), -1)
        pts5_t_inv = t_inv(pts5_l, (1, 0, 0), -1)
        ind5 = set_int_ind(pt, pts5)
        pts_cube[ind5, :] = pts5_t_inv

    if len(pts6) > 0:
        pts6_l = lamb_area(pts6, (0, 1, 0), -1)
        pts6_t_inv = t_inv(pts6_l, (0, 1, 0), -1)
        ind6 = set_int_ind(pt, pts6)
        pts_cube[ind6, :] = pts6_t_inv

    return pts_cube

def cube2sphr(pt, tol=1e-12):

    pts = np.copy(pt)
    beta = math.sqrt(np.pi/6)

    cond_1 = np.abs(pts[:, 2] - beta) <= tol
    pts1 = pts[cond_1]; pts = np.delete(pts, np.where(cond_1)[0], 0)

    cond_2 = np.abs(pts[:, 0] - beta) <= tol
    pts2 = pts[cond_2]; pts = np.delete(pts, np.where(cond_2)[0], 0)

    cond_3 = np.abs(pts[:, 1] - beta) <= tol
    pts3 = pts[cond_3]; pts = np.delete(pts, np.where(cond_3)[0], 0)

    cond_4 = np.abs(pts[:, 2] + beta) <= tol
    pts4 = pts[cond_4]; pts = np.delete(pts, np.where(cond_4)[0], 0)

    cond_5 = np.abs(pts[:, 0] + beta) <= tol
    pts5 = pts[cond_5]; pts = np.delete(pts, np.where(cond_5)[0], 0)

    cond_6 = np.abs(pts[:, 1] + beta) <= tol
    pts6 = pts[cond_6]; pts = np.delete(pts, np.where(cond_6)[0], 0)

    if len(pts) != 0:
        raise " Algorithm is incorrect!"

    pts_sphr = np.zeros(np.shape(pt))
    if len(pts1) > 0:
        pts1_t = t(pts1, (0, 0, 1), 1)
        pts1_l_inv = lamb_area_inv(pts1_t, (0, 0, 1), 1)
        ind1 = set_int_ind(pt, pts1)
        pts_sphr[ind1, :] = pts1_l_inv

    if len(pts2) > 0:
        pts2_t = t(pts2, (1, 0, 0), 1)
        pts2_l_inv = lamb_area_inv(pts2_t, (1, 0, 0), 1)
        ind2 = set_int_ind(pt, pts2)
        pts_sphr[ind2, :] = pts2_l_inv

    if len(pts3) > 0:
        pts3_t = t(pts3, (0, 1, 0), 1)
        pts3_l_inv = lamb_area_inv(pts3_t, (0, 1, 0), 1)
        ind3 = set_int_ind(pt, pts3)
        pts_sphr[ind3, :] = pts3_l_inv

    if len(pts4) > 0:
        pts4_t = t(pts4, (0, 0, 1), -1)
        pts4_l_inv = lamb_area_inv(pts4_t, (0, 0, 1), -1)
        ind4 = set_int_ind(pt, pts4)
        pts_sphr[ind4, :] = pts4_l_inv

    if len(pts5) > 0:
        pts5_t = t(pts5, (1, 0, 0), -1)
        pts5_l_inv = lamb_area_inv(pts5_t, (1, 0, 0), -1)
        ind5 = set_int_ind(pt, pts5)
        pts_sphr[ind5, :] = pts5_l_inv

    if len(pts6) > 0:
        pts6_t = t(pts6, (0, 1, 0), -1)
        pts6_l_inv = lamb_area_inv(pts6_t, (0, 1, 0), -1)
        ind6 = set_int_ind(pt, pts6)
        pts_sphr[ind6, :] = pts6_l_inv

    return pts_sphr

def lamb_area(pts, plane, pl_type):

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

    if pl_type == 1:
        xl = np.multiply(np.sqrt(np.true_divide(2, (1+z))), x)
        yl = np.multiply(np.sqrt(np.true_divide(2, (1+z))), y)
        zl = np.zeros(np.shape(z)[0]); zl.fill(1.0)
    if pl_type == -1:
        xl = np.multiply(np.sqrt(np.true_divide(2, (1-z))), x)
        yl = np.multiply(np.sqrt(np.true_divide(2, (1-z))), y)
        zl = np.zeros(np.shape(z)[0]); zl.fill(-1.0)

    if plane == (0, 0, 1):
        pts_l = np.column_stack((xl, yl, zl))
    elif plane == (0, 1, 0):
        pts_l = np.column_stack((yl, zl, xl))
    elif plane == (1, 0, 0):
        pts_l = np.column_stack((zl, xl, yl))

    return pts_l

def t_inv(pts, plane, pl_type):

    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]

    cond1 = np.abs(y) <= np.abs(x)
    x1 = x[cond1]
    y1 = y[cond1]
    ind1 = np.where(cond1)[0]

    cond2 = np.abs(x) < np.abs(y)
    x2 = x[cond2]
    y2 = y[cond2]
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
    elif np.shape(x2)[0] == 0:
        xt_inv = xt_1
        yt_inv = yt_1
    else:
        xt_inv = np.zeros(np.shape(x))
        yt_inv = np.zeros(np.shape(y))

        xt_inv[ind1] = xt_1
        xt_inv[ind2] = xt_2
        yt_inv[ind1] = yt_1
        yt_inv[ind2] = yt_2

    if pl_type == 1:
        zt_inv = np.empty(np.shape(xt_inv)[0])
        zt_inv.fill(beta)
    if pl_type == -1:
        zt_inv = np.empty(np.shape(xt_inv)[0])
        zt_inv.fill(-beta)

    if plane == (0, 0, 1):
        pts_t_inv = np.column_stack((xt_inv, yt_inv, zt_inv))
    elif plane == (0, 1, 0):
        pts_t_inv = np.column_stack((yt_inv, zt_inv, xt_inv))
    elif plane == (1, 0, 0):
        pts_t_inv = np.column_stack((zt_inv, xt_inv, yt_inv))

    return pts_t_inv

def t(pts, plane, pl_type):

    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]

    cond1 = np.abs(y) <= np.abs(x)
    x1 = x[cond1]
    y1 = y[cond1]
    ind1 = np.where(cond1)[0]

    cond2 = np.abs(x) < np.abs(y)
    x2 = x[cond2]
    y2 = y[cond2]
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
    elif np.shape(x2)[0] == 0:
        xt = xt_1
        yt = yt_1
    else:
        xt = np.zeros(np.shape(x))
        yt = np.zeros(np.shape(y))

        xt[ind1] = xt_1
        xt[ind2] = xt_2
        yt[ind1] = yt_1
        yt[ind2] = yt_2


    if pl_type == 1:
        zt = np.empty(np.shape(xt)[0])
        zt.fill(1.0)
    if pl_type == -1:
        zt = np.empty(np.shape(xt)[0])
        zt.fill(-1.0)

    if plane == (0, 0, 1):
        pts_t = np.column_stack((xt, yt, zt))
    elif plane == (0, 1, 0):
        pts_t = np.column_stack((yt, zt, xt))
    elif plane == (1, 0, 0):
        pts_t = np.column_stack((zt, xt, yt))

    return pts_t

def lamb_area_inv(pts, plane, pl_type):

    if plane == (0, 0, 1):
        x = pts[:, 0]
        y = pts[:, 1]
    elif plane == (0, 1, 0):
        x = pts[:, 2]
        y = pts[:, 0]
    elif plane == (1, 0, 0):
        x = pts[:, 1]
        y = pts[:, 2]

    xl = ((1.0 - (x*x + y*y)/4.0)**0.5) * x
    yl = ((1.0 - (x*x + y*y)/4.0)**0.5) * y
    zl = 1.0 - (x*x + y*y)/2.0

    if pl_type == -1:
        zl = -zl

    if plane == (0, 0, 1):
        pts_l_inv = np.column_stack((xl, yl, zl))
    elif plane == (0, 1, 0):
        pts_l_inv = np.column_stack((yl, zl, xl))
    elif plane == (1, 0, 0):
        pts_l_inv = np.column_stack((zl, xl, yl))

    return pts_l_inv

def set_int_ind(set1, set2):

    set1_v = set1.view([('', set1.dtype)] * set1.shape[1]).ravel()
    set2_v = set2.view([('', set2.dtype)] * set2.shape[1]).ravel()

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

    pts_cube = sphr2cube(sp)
    cp = np.copy(pts_cube)
    pts_sphr = cube2sphr(cp)
    print sp, '\n\n', pts_cube, '\n\n', pts_sphr
    print '\n\n', sp - pts_sphr, '\n\n', np.max(sp-pts_sphr)
    return sp
# check2(100)