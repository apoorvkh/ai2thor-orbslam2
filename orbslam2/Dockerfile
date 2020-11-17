FROM ai2thor-docker:latest

RUN rm /app/example_agent.py /app/install_nvidia.sh /app/requirements.txt
RUN apt update

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
    cmake -D CMAKE_BUILD_TYPE=Release .. && make -j"$(nproc)" && make install && \
    cd ../.. && \
    rm -rf 3.4.12.zip opencv-3.4.12

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip && unzip eigen-3.3.8.zip && \
    cd eigen-3.3.8 && \
    mkdir build && cd build && \
    cmake .. && make install && \
    cd ../.. && \
    rm -rf eigen-3.3.8.zip eigen-3.3.8

RUN pip3 install Cython && apt-get remove python3-apt && apt-get install python3-apt libcanberra-gtk*
ENV LD_LIBRARY_PATH "/app/ORB_SLAM2/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

RUN git clone https://github.com/apoorvkh/ORB_SLAM2.git && cd ORB_SLAM2 && \
    chmod +x build.sh && ./build.sh && \
    python3 setup.py build_ext --inplace && \
    cd ..

ENV PYTHONPATH "/app/ORB_SLAM2:/app/ai2thor_docker:${PYTHONPATH}"

COPY ai2thor_orbslam2.py /app
COPY demo /app/demo/
