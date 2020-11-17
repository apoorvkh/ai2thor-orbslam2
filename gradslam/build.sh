cd .. && bash ./build_ai2thor_docker.sh && cd gradslam

docker build -t ai2thor-gradslam:latest .
