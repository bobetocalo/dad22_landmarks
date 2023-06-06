# Landmark Detection using DAD-3DHeads CVPR (2022)

#### Requisites
- images_framework https://github.com/bobetocalo/images_framework
- dlib

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
usage: Dad22Landmarks [--pose POSE]
```

* Use the --pose option to choose pose index for simultaneous train.
```
> python images_framework/alignment/dad22_landmarks/test/dad22_landmarks_test.py --input-data images_framework/alignment/dad22_landmarks/test/example.tif --database aflw --save-image
```

