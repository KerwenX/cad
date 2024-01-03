import numpy as np
from trimesh import load as load_model, viewer
from scan2cad_rasterizer import Rasterizer
import matplotlib.pyplot as plt
import getRotation,getTranslation,getScale
def make_M_from_tqs(pose, s: list, center=None) -> np.ndarray:
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = pose.dot(S).dot(C)
    return M


if __name__ == '__main__':
    intr =np.asarray([
            [435.19*4,    0.,  239.9*2,   0.],
            [  0.,  435.19*4,  179.91*2, 0.  ],
            [  0.,    0.,    1.,    0.  ],
            [  0.,    0.,    0.,    1.  ]]
    )# 720 * 960

    Rotation = getRotation.get_rotation(0,-40,180)
    Translation =  getTranslation.get_translation(0,0,3)
    Scale = getScale.get_scale(1,1,1)
    pose = np.eye(4)
    pose[:3,:3] = Rotation
    pose[:3,3] = Translation
    pose = pose.dot(Scale)
    print(pose)

    cad_path = 'camera.obj'
    mesh = load_model(cad_path,file_type='obj',force='mesh')

    raster = Rasterizer(intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], False, True)

    mesh.apply_transform(pose)
    raster.add_model(
        np.asarray(mesh.faces, dtype=raster.index_dtype),
        np.asarray(mesh.vertices, dtype=raster.scalar_dtype),
        1,
        np.asarray(mesh.face_normals, raster.scalar_dtype)
    )
    raster.rasterize()
    instances = np.uint8(raster.read_idx())
    colors = np.unique(instances)
    if len(colors)==1:
        raster.clear_models()
        print("visual None ! ")
        exit(-1)
    raster.set_colors({7: np.array([0, 0.9, 0.9]), 2: np.array([0, 0.8, 1])})
    raster.render_colors(0.25)
    shaded = raster.read_color()
    rows,cols,_ = np.where(shaded != 0)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    cropped_image = shaded[min_row:max_row + 1, min_col:max_col + 1, :]
    alpha = np.where(np.all(cropped_image==0,axis=-1),0,1)
    rgba = np.concatenate((cropped_image,alpha[:,:,np.newaxis]),axis=-1)
    plt.figure()
    plt.imshow(rgba)
    plt.axis('off')
    # 设置图像背景色为透明
    # plt.savefig(cad_output_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.show()
    plt.close('all')

    raster.clear_models()
