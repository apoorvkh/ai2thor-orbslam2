bash ./build_ai2thor_docker.sh && cd orbslam2
docker build -t ai2thor-orbslam2:latest .

xhost +local:root
docker run --privileged --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it ai2thor-orbslam2:latest bash