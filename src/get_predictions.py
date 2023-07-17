from functools import partial
import json
from collections import namedtuple
import os
import numpy as np
from fire import Fire
from pytorch_toolbelt.utils import read_rgb_image
from dad_3dheads_benchmark.utils import get_68_landmarks, get_7_landmarks_from_68
from model_training.model.flame import rotation_mat_from_flame_params
from predictor import FaceMeshPredictor
from demo_utils import (
    draw_landmarks,
    draw_3d_landmarks,
    draw_mesh,
    draw_pose,
    get_mesh,
    get_flame_params,
    get_flame_params2,
    get_output_path,
    MeshSaver,
    ImageSaver,
    JsonSaver,
)

DemoFuncs = namedtuple(
    "DemoFuncs",
    ["processor", "saver"],
)

demo_funcs = {
    "68_landmarks": DemoFuncs(draw_landmarks, ImageSaver),
    "191_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="191"), ImageSaver),
    "445_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="445"), ImageSaver),
    "head_mesh": DemoFuncs(partial(draw_mesh, subset="head"), ImageSaver),
    "face_mesh": DemoFuncs(partial(draw_mesh, subset="face"), ImageSaver),
    "pose": DemoFuncs(draw_pose, ImageSaver),
    "3d_mesh": DemoFuncs(get_mesh, MeshSaver),
    "flame_params": DemoFuncs(get_flame_params, JsonSaver)
}





def get_predictions(
    input_image_path: str = "/home/database8TB/dad-3dheads/val/images", #'images/demo_heads/1.jpeg',
    outputs_folder: str = "outputs",
    type_of_output: str = "flame_params",
) -> None:
    
    os.makedirs(outputs_folder, exist_ok=True)
    folder_path = "/home/database8TB/dad-3dheads/val/images"
    file_list = os.listdir(folder_path)
    # Preprocess and get predictions.
    data = {}
    for image in file_list:
        image_path = os.path.join(folder_path, image)
        id_image = image_path.replace("/home/database8TB/dad-3dheads/val/images/","")
        id_image = id_image.replace(".png","")
        print("ID_IMAGE",id_image)
        image = read_rgb_image(image_path)
        predictor = FaceMeshPredictor.dad_3dnet()
        predictions = predictor(image)
        points = predictions['points']
        td_vertices = predictions['3d_vertices']
        tdmm_params = predictions['3dmm_params']
        full_gt_lmks = get_68_landmarks(td_vertices)
        svn_lmks = get_7_landmarks_from_68(full_gt_lmks)
        flame_params = get_flame_params2(predictions)
        rot_mat = rotation_mat_from_flame_params(flame_params)
        images = {
                id_image: {
                    '68_landmarks_2d': points.tolist(),
                    'N_landmarks_3d': td_vertices.tolist(),
                    '7_landmarks_3d': svn_lmks.tolist(),
                    'rotation_matrix': rot_mat.tolist()
                }
        }
        data.update(images)
    with open('predictions.json', 'w') as archivo:
        json.dump(data, archivo)
 
if __name__ == "__main__":
    Fire(get_predictions)
