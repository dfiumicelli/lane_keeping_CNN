#!/bin/bash

# Abilita X11 forwarding per grafica
xhost +local:docker

# Run container con display
docker run -it --rm \
    --privileged \
    --network host \
    -e ROS_DOMAIN_ID=30 \
    -e ROS_LOCALHOST_ONLY=0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    lane_keeping_cnn_img
