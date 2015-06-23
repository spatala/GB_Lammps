import numpy as np
import math
import random as rnd 

def generate_fz_location(fz_loc):
    x_tol = 1e-10; max_k = 2

    k = (math.sqrt(2)-1); k1 = 1/(math.sqrt(1+2*k*k))

    if fz_loc == 'Pt_O':
        q0 = 1; q1 = 0; q2 = 0; q3 = 0
        cond0 = abs(q0-1) <= x_tol
        if cond0:
            mis_quat = [q0, q1, q2, q3]
            return mis_quat

    elif fz_loc == 'Pt_A':
        q0 = math.cos(np.pi/8); q1 =0; q2 = 0; q3 = math.sin(np.pi/8)
        cond0 = abs(q0 - math.cos(np.pi/8)) <= x_tol
        cond1 = abs(q1) <= x_tol
        cond2 = abs(q2) <= x_tol
        cond3 = abs(q3 - math.sin(np.pi/8)) <= x_tol
        if cond0 and cond1 and cond2 and cond3:
            mis_quat = [q0, q1, q2, q3]
            return mis_quat

    elif fz_loc == 'Pt_E':
        q0 = math.sqrt(3)/2; q1 = 1/(2*math.sqrt(3))
        q2 = 1/(2*math.sqrt(3)); q3 = 1/(2*math.sqrt(3))
        cond0 = abs(q0 - math.sqrt(3)/2) <= x_tol
        cond1 = abs(q1 - 1/(2*math.sqrt(3))) <= x_tol
        cond2 = abs(q2 - 1/(2*math.sqrt(3))) <= x_tol
        cond3 = abs(q3 - 1/(2*math.sqrt(3))) <= x_tol
        if cond0 and cond1 and cond2 and cond3:
            mis_quat = [q0, q1, q2, q3]
            return mis_quat

    elif fz_loc == 'Pt_C':
        q0 = 1/(k*2*math.sqrt(2)); q1 = 1/(2*math.sqrt(2))
        q2 = 1/(2*math.sqrt(2)); q3 = k/(2*math.sqrt(2))
        cond0 = abs(q1 - 1/(k*2*math.sqrt(2))) <= x_tol
        cond1 = abs(q1 - 1/(2*math.sqrt(2))) <= x_tol
        cond2 = abs(q2 - 1/(2*math.sqrt(2))) <= x_tol
        cond3 = abs(q3 - 1/(2*math.sqrt(2))) <= x_tol
        if cond0 and cond1 and cond2 and cond3:
            mis_quat = [q0, q1, q2, q3]
            return mis_quat

    elif fz_loc == 'Line_OA':


