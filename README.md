# Landmark Detection using DAD-3DHeads CVPR (2022)

#### Requisites
- images_framework https://github.com/pcr-upm/images_framework
- Pytorch (1.9.0)
- Pytorch-toolbelt (0.5.0)
- Hydra (1.1.0)
- SMPL-X (0.1.26)
- Chumpy (0.70)

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── alignment
        └── dad22_landmarks
```
#### Usage
```
usage: dad22_landmarks_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: Dad22Landmarks [--gpu GPU] [--cfg CFG]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
* Use the --cfg option to set the experiment configure file name.
```
> python images_framework/alignment/dad22_landmarks/test/dad22_landmarks_test.py --input-data images_framework/alignment/dad22_landmarks/test/example.tif --database dad --gpu 0 --save-image
```
