xhost +local:root
docker run --privileged --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it ai2thor-gradslam:latest bash
xhost -local:root
