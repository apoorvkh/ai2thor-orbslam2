FROM ai2thor-docker:latest

RUN cd /app && rm example_agent.py install_nvidia.sh requirements.txt
COPY ai2thor_gradslam.py /app
COPY demo /app/demo/

RUN apt update

RUN pip3 install "torch>=1.6.0"

RUN git clone https://github.com/krrish94/chamferdist.git && cd chamferdist && \
    pip3 install . && \
    cd /app

RUN git clone https://github.com/gradslam/gradslam.git && cd gradslam && \
    pip3 install -e . && \
    cd /app

ENV PYTHONPATH "/app/ai2thor_docker:${PYTHONPATH}"
