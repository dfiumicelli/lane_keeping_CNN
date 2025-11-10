FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-image-transport \
    ros-humble-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# FIX: Installa NumPy 1.x PRIMA di onnxruntime
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    onnxruntime \
    opencv-python \
    pyyaml

RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

COPY src/line_follower_CNN/package.xml ./src/line_follower_CNN/

RUN apt-get update && \
    rosdep update && \
    rosdep install -r --from-paths src -i -y --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

COPY src ./src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/ros/humble/setup.bash\n\
source /ros2_ws/install/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
