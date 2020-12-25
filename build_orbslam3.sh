bash ./build_ai2thor_docker.sh && cd orbslam3
docker build -t ai2thor-orbslam3:latest .

xhost +local:root
docker run --privileged --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it ai2thor-orbslam3:latest bash