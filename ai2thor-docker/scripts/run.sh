#!/bin/bash
ALFREDPATH="$( cd -- "$(dirname "$0")"/../.. >/dev/null 2>&1 ; pwd -P )"
echo $ALFREDPATH
if [ -z $1 ];
then
    EXTRA=""
else
    EXTRA="-v $1"
fi
docker run --rm -it --privileged -v $ALFREDPATH:/home/$USER/alfred -v ~/.ssh:/home/$USER/.ssh -v ~/.torch:/home/$USER/.torch $EXTRA ai2thor-docker bash --user $USER
