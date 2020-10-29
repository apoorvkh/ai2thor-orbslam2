if [[ "$(docker images -q ai2thor-docker:latest 2> /dev/null)" == "" ]]; then
    git clone git@github.com:allenai/ai2thor-docker.git
    cd ai2thor-docker
    ./scripts/build.sh
    cd ..
fi

docker build -t ai2thor-orbslam2:latest .
