# ai2thor-slam

Prototyping and integrations of different SLAM algorithms with [ai2thor](https://ai2thor.allenai.org/).

Currently supports:
- [gradslam](https://github.com/gradslam/gradslam)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) ([via my fork](https://github.com/apoorvkh/ORB_SLAM2))
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) ([via my fork](https://github.com/apoorvkh/ORB_SLAM3))

Currently, ORB-SLAM2 seems to produce the most accurate trajectory for the ai2thor environment (see demo).

## Requirements:
- Docker
- nvidia driver and compatible CUDA version

## Building container
```
export SLAM=gradslam # or orbslam2, orbslam3

# Building ai2thor-docker image
if [[ "$(docker images -q ai2thor-docker:latest 2> /dev/null)" == "" ]]; then
    git clone git@github.com:allenai/ai2thor-docker.git
    cd ai2thor-docker
    ./scripts/build.sh
    cd ..
fi

# Building ai2thor + SLAM image
bash ./build_ai2thor_docker.sh && cd $SLAM
docker build -t ai2thor-$SLAM:latest .
```

## Starting the container
```
export SLAM=gradslam # or orbslam2, orbslam3

docker run --privileged --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it ai2thor-$SLAM:latest bash
```

## Running demo

Outside of Docker container, run `xhost +local:root`

In Docker container, run `python3 demo/example_agent.py`