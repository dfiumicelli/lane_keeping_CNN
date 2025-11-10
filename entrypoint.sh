#!/bin/bash
set -e

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Source workspace
source /ros2_ws/install/setup.bash

# Export variabili ROS2
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Se passati argomenti, eseguili
if [ $# -eq 0 ]; then
    # Nessun argomento → bash interattiva
    exec bash
else
    # Argomenti passati → esegui
    exec "$@"
fi
