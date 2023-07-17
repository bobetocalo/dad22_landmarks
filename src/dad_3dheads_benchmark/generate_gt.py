import json
from tqdm import tqdm
from fire import Fire
from utils import read_img


def generate_gt(
        #base_path: str,
        output_filename: str = 'data/ground_truth_val_all.json',
):
    with open(f'/home/database8TB/dad-3dheads/val/val.json', "r") as f:
        val_anno = json.load(f)

    val_json = []
    list_img = ["2fde5bd6-08ed-44c5-bf68-01514f0ea76a","ce518d9f-6c95-459d-82f6-68df64652c61",
    "e720602a-0f0b-4563-8fa6-506e969c2aa3","be0a2478-a69e-4dc7-8117-c7fb2e6e7d45","8eaf047f-f7ef-4a25-83ef-040b731a253b"] #,"fffedfdf-94e3-42f5-bb86-15b066a0af3f", "8020f2b6-7679-4dc5-8ab6-11de9b8f9205", "fffefe60-6a19-416e-94e7-bd22352f50be","fffe18f0-084c-4e0e-a1a0-717f5033740c"]
    #count = 0
    for el in tqdm(val_anno):
        #if el['item_id'] not in list_img : continue
        #print(el['item_id'])
        annotation_path = f'/home/database8TB/dad-3dheads/val/annotations/{el["item_id"]}.json'
        image = read_img(f'/home/database8TB/dad-3dheads/val/images/{el["item_id"]}.png')
        image_height = image.shape[0]
        anno = json.loads(open(annotation_path).read())
        val_json.append(
            {
                'id': el['item_id'],
                'bbox': el['bbox'],
                'vertices': anno['vertices'],
                'model_view_matrix': anno['model_view_matrix'],
                'projection_matrix': anno['projection_matrix'],
                'image_height': image_height
            }
        )
        #count = count + 1
    with open(output_filename, "w") as out:
        json.dump(val_json, out)

    #print(count)
if __name__ == "__main__":
    Fire(generate_gt)
