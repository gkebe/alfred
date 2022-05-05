#!/bin/bash
ALFREDPATH="$( cd -- "$(dirname "$0")"/../.. >/dev/null 2>&1 ; pwd -P )"
echo $ALFREDPATH
if [ -z $1 ];
then
    EXTRA=""
else
    EXTRA="-v $1"
fi
docker run --rm -it --privileged -v $ALFREDPATH:/home/iral/alfred -v ~/.ssh:/home/iral/.ssh -v ~/.torch:/home/iral/.torch -e ALFRED_ROOT=/home/iral/alfred -e HOME=/home/iral -p 8888:8888 --user iral --ipc=host --network=host -v /usr/bin/nvidia-xconfig:/usr/bin/nvidia-xconfig --gpus all  $EXTRA ai2thor-docker bash
