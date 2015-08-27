import numpy as np
import math
import GBpy.quaternion as GBq
import scipy.io as sio

def misorient_fz_432(misquats, tol = 1e-14):

    quat = np.zeros((24, 4))

    k1 = 1/math.sqrt(2); k2 = 0.5

    quat[0, :] = [1, 0, 0, 0]
    quat[1, :] = [0, 1, 0, 0]
    quat[2, :] = [0, 0, 1, 0]
    quat[3, :] = [0, 0, 0, 1]

    quat[4, :] = [k1, k1, 0, 0]
    quat[5, :] = [k1, -k1, 0, 0]

    quat[6, :] = [k1, 0, k1, 0]
    quat[7, :] = [k1, 0, -k1, 0]

    quat[8, :] = [k1, 0, 0, k1]
    quat[9, :] = [k1, 0, 0, -k1]

    quat[10, :] = [0, k1, k1, 0]
    quat[11, :] = [0, -k1, k1, 0]

    quat[12, :] = [0, k1, 0, k1]
    quat[13, :] = [0, k1, 0, -k1]

    quat[14, :] = [0, 0, k1, k1]
    quat[15, :] = [0, 0, -k1, k1]

    quat[16, :] = [k2, k2, k2, k2]
    quat[17, :] = [k2, -k2, k2, k2]
    quat[18, :] = [k2, k2, -k2, k2]
    quat[19, :] = [k2, k2, k2, -k2]
    quat[20, :] = [k2, k2, -k2, -k2]
    quat[21, :] = [k2, -k2, k2, -k2]
    quat[22, :] = [k2, -k2, -k2, k2]
    quat[23, :] = [k2, -k2, -k2, -k2]

    disquats = np.zeros(np.shape(misquats))
    disquats.fill(float('NaN'))
    try:
        rng1 = np.shape(misquats)[1]
    except:
        rng1 = 1

    for ct1 in range(rng1):
        misquats1 = np.array(misquats[ct1:, ]); t_found = 0
        if t_found ==0:
            for j in range(np.shape(quat)[0]):
                # print j
                temp_quat1 = GBq.mtimes(GBq.Quaternion(misquats1), GBq.Quaternion(quat[j, :]))
                if t_found ==0:
                    for k in range(np.shape(quat)[0]):
                        temp_quat2 = GBq.mtimes(GBq.Quaternion(quat[k, :]), GBq.Quaternion(temp_quat1))
                        if t_found ==0:
                            temp_quat3 = temp_quat2

                            q0 = temp_quat3[0]; q1 = temp_quat3[1]; q2 = temp_quat3[2]; q3 = temp_quat3[3]
                            if check_cond(q0, q1, q2, q3, tol):
                                disquats[ct1 :] = [q0, q1, q2, q3]; t_found=1
                                break

                            q0 = -temp_quat3[0]; q1 = -temp_quat3[1]; q2 = -temp_quat3[2]; q3 = -temp_quat3[3]
                            if check_cond(q0, q1, q2, q3, tol):
                                disquats[ct1 :] = [q0, q1, q2, q3]; t_found=1
                                break

                            q0 = temp_quat3[0]; q1 = -temp_quat3[1]; q2 = -temp_quat3[2]; q3 = -temp_quat3[3]
                            if check_cond(q0, q1, q2, q3, tol):
                                disquats[ct1 :] = [q0, q1, q2, q3];t_found=1
                                break

                            q0 = -temp_quat3[0]; q1 = temp_quat3[1]; q2 = temp_quat3[2]; q3 = temp_quat3[3]
                            if check_cond(q0, q1, q2, q3, tol):
                                disquats[ct1, :] = [q0, q1, q2, q3]; t_found=1
                                break
    return disquats
def check_cond(q0, q1, q2, q3, tol):
    t_found =0
    cond1 = q1 -q2 > -tol
    cond2 = q2 -q3 > -tol
    cond3 = q3 > -tol
    cond4 = q0 -(math.sqrt(2)+1)*q1 > -tol
    cond5 = q0 -(q1 + q2 + q3) > -tol

    if cond1 and cond2 and cond3 and cond4 and cond5:
        cond6 = abs(q0 - (math.sqrt(2)+1)*2) <= 0
        if cond6:
            cond7 = q2 - (math.sqrt(2)+1)*q3 <= tol
            if cond7:
                t_found = 1
                return t_found
        t_found = 1
        return t_found
    return t_found

# contents = sio.loadmat('mfz432.mat')
# misquats = contents['Misquats'][0]
# misorient_fz_432(misquats)






