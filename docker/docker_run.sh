#!/bin/bash -e

commands="$1"

repo_dir=$(cd $(dirname $0); pwd)/..

docker run -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v ${repo_dir}:/workspace \
    --rm \
    nn_with_c:${USER} ${commands}
