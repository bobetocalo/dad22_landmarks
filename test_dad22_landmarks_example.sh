#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t dad22_landmarks_image .
sudo docker volume create --name dad22_landmarks_volume
sudo docker run --name dad22_landmarks_container -v dad22_landmarks_volume:/home/username --rm --gpus all -it -d dad22_landmarks_image bash
sudo docker exec -w /home/username/ dad22_landmarks_container python images_framework/alignment/dad22_landmarks/test/dad22_landmarks_test.py --input-data images_framework/alignment/dad22_landmarks/test/example.tif --database dad --gpu 0 --save-image
sudo docker stop dad22_landmarks_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/dad22_landmarks_volume/_data/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
