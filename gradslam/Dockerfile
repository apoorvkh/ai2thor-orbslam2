FROM ai2thor-docker:latest

RUN rm /app/example_agent.py /app/install_nvidia.sh /app/requirements.txt
RUN apt update

RUN pip3 install "torch>=1.6.0"

RUN git clone https://github.com/gradslam/gradslam.git && cd gradslam && \
    pip3 install -e . && \
    cd ..

ENV PYTHONPATH "/app/ai2thor_docker:${PYTHONPATH}"

COPY ai2thor_gradslam.py /app
COPY demo /app/demo/
