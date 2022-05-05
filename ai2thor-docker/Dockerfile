ARG CUDA_VERSION
FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04

RUN apt-key del "7fa2af80" \
&& export this_distro="$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')" \
&& export this_version="$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')" \
&& apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${this_distro}${this_version}/x86_64/3bf863cc.pub"

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
RUN apt-get -y install python3-pip libvulkan1 python3-virtualenv python3-dev vim pciutils wget git kmod vim git xorg screen nano

WORKDIR /home/$USER_NAME
RUN cd $WORKDIR

RUN mkdir /home/$USER_NAME/alfred
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .
copy ./scripts/install_nvidia.sh /home/$USER_NAME/
RUN NVIDIA_VERSION=$NVIDIA_VERSION /home/$USER_NAME/install_nvidia.sh

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh \
    && bash Miniconda3-py37_4.11.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && PATH='/root/miniconda3/bin:$PATH' \
    && printf "PATH='/root/miniconda3/bin:$PATH'" >> ${WORKDIR}/.bashrc \
    && conda init

ENTRYPOINT bash -c "export ALFRED_ROOT=~/alfred && PATH="/root/miniconda3/bin:$PATH" && conda init && /bin/bash && echo -e "# ~/.screenrc\nshell -/bin/bash" >> ~/.screenrc"
