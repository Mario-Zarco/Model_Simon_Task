import numpy as np


def get_angle_2d(pi, pn, pf):
    if pf[0] < 0:
        v2 = pn - pi
        v1 = pf - pi
    else:
        v1 = pn - pi
        v2 = pf - pi

    theta = np.arctan2(np.linalg.det(np.row_stack((v2, v1))), np.dot(v1, v2))

    return theta


def maximum_deviation_2d(x_tr, y_tr):
    xy = np.column_stack((x_tr, y_tr))
    pi = xy[0, :]
    pf = xy[-1, :]
    p_tr = xy[1:-1, :]

    dis = []
    ang = []
    for pn in p_tr:
        theta = get_angle_2d(pi, pn, pf)
        d = np.linalg.norm(pn - pi) * np.sin(np.abs(theta))
        dis = np.append(dis, d)
        ang = np.append(ang, theta)

    max_dis = np.max(dis)
    max_dev = max_dis / np.linalg.norm(pf - pi)
    idx = np.argwhere(max_dis == dis)[0][0]

    x_max = p_tr[idx, 0]
    y_max = p_tr[idx, 1]

    return np.sign(ang[idx]) * max_dev, np.sign(ang[idx]) * max_dis, x_max, y_max