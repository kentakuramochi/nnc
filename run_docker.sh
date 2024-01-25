#!/bin/bash

docker_image_name=scnn

if [ "$(docker images | grep ${docker_image_name})" == "" ]; then
    docker build -t ${docker_image_name} .
fi

docker run -u $(id -u $USER):$(id -g $USER) \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v $(pwd):/workspace \
    --rm \
    -it ${docker_image_name} /bin/bash
