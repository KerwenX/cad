import numpy as np

def get_scale(x:float,y:float,z:float):
    # intput:
    #   x,y,z
    # output:
    #   S:4*4

    S = np.eye(4)
    S[:3,:3] = np.diag([x,y,z])
    return S