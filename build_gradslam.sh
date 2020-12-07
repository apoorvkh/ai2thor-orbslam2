cd .. && bash ./build_ai2thor_docker.sh && cd gradslam
docker build -t ai2thor-gradslam:latest .

xhost +local:root
docker run --privileged --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it ai2thor-gradslam:latest bash
