FROM ai2thor-docker:latest

RUN apt install libgl1-mesa-dev libglew-dev cmake && \
    git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. && cmake --build . && make install && \
    cd ../.. && \
    rm -rf Pangolin

RUN apt install unzip libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy && \
    wget https://github.com/opencv/opencv/archive/3.4.12.zip && unzip 3.4.12.zip && \
    cd opencv-3.4.12 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && make -j8 && make install && \
    cd ../.. && \
    rm -rf 3.4.12.zip opencv-3.4.12

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip && unzip eigen-3.3.8.zip && \
    cd eigen-3.3.8 && \
    mkdir build && cd build && \
    cmake .. && make install && \
    cd ../.. && \
    rm -rf eigen-3.3.8.zip eigen-3.3.8

RUN git clone https://github.com/apoorvkh/ORB_SLAM2.git && \
    cd ORB_SLAM2 && \
    chmod +x build.sh && .\build.sh && \
    cd ..
