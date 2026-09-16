#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t dad22_landmarks_image .
sudo docker run --name dad22_landmarks_container --rm --gpus all -it -d dad22_landmarks_image bash
sudo docker exec -w /home/username/dad22_landmarks dad22_landmarks_container python test/dad22_landmarks_test.py --input-data test/example.tif --database dad --gpu 0 --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp dad22_landmarks_container:/home/username/conda/envs/dad22/lib/python3.8/site-packages/pcr_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f dad22_landmarks_container
sudo docker image rm dad22_landmarks_image
sudo docker builder prune -a -f