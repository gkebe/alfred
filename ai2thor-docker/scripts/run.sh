#!/bin/bash
ALFREDPATH="$( cd -- "$(dirname "$0")"/../.. >/dev/null 2>&1 ; pwd -P )"
echo $ALFREDPATH
docker run --rm -it --privileged -v $ALFREDPATH:/home/$USER/alfred -v ~/.ssh:/home/$USER/.ssh -v ~/.torch:/home/$USER/.torch ai2thor-docker bash --user $USER
