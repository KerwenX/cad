import numpy as np

def get_rotation(theta_x:int, theta_y:int, theta_z:int):

    # 将角度转换为弧度
    roll_rad = np.radians(theta_x)
    pitch_rad = np.radians(theta_y)
    yaw_rad = np.radians(theta_z)

    # 计算旋转矩阵
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad), np.cos(roll_rad)]])

    # 旋转矩阵相乘得到最终的旋转矩阵
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    return R


def rotation_matrix_to_quaternion(R):
    # R : 3*3
    trace = np.trace(R)
    if trace > 0:
        w = np.sqrt(1 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            w = (R[2, 1] - R[1, 2]) / (4 * x)
            y = (R[0, 1] + R[1, 0]) / (4 * x)
            z = (R[0, 2] + R[2, 0]) / (4 * x)
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            w = (R[0, 2] - R[2, 0]) / (4 * y)
            x = (R[0, 1] + R[1, 0]) / (4 * y)
            z = (R[1, 2] + R[2, 1]) / (4 * y)
        else:
            z = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            w = (R[1, 0] - R[0, 1]) / (4 * z)
            x = (R[0, 2] + R[2, 0]) / (4 * z)
            y = (R[1, 2] + R[2, 1]) / (4 * z)

    q = np.quaternion(w, x, y, z)
    return q