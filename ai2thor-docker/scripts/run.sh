#!/bin/bash
ALFREDPATH="$( cd -- "$(dirname "$0")"/../.. >/dev/null 2>&1 ; pwd -P )"
echo $ALFREDPATH
if [ -z $1 ];
then
    EXTRA=""
else
    EXTRA="-v $1"
fi
docker run --rm -it --privileged -v $ALFREDPATH:/home/$USER/alfred -v ~/.ssh:/home/$USER/.ssh -v ~/.torch:/home/$USER/.torch -e ALFRED_ROOT=/home/$USER/alfred -e HOME=/home/$USER -p 8888:8888 --user $USER --ipc=host --network=host -v /usr/bin/nvidia-xconfig:/usr/bin/nvidia-xconfig --gpus all  $EXTRA ai2thor-docker bash
