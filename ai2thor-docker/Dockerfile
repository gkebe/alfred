ARG CUDA_VERSION

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04
ARG NVIDIA_VERSION

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip libvulkan1 python3-virtualenv python3-dev vim pciutils wget git kmod vim git xorg screen

WORKDIR /home/$USER_NAME
RUN cd $WORKDIR
ENV VIRTUAL_ENV=/home/$USER_NAME/alfred_env
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python requirements
RUN pip install --upgrade pip==19.3.1
RUN pip install -U setuptools
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir /home/$USER_NAME/alfred
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .
copy ./scripts/install_nvidia.sh /home/$USER_NAME/
RUN NVIDIA_VERSION=$NVIDIA_VERSION /home/$USER_NAME/install_nvidia.sh


ENTRYPOINT bash -c "export ALFRED_ROOT=~/alfred && /bin/bash"
