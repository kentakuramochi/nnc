#!/bin/bash -eu

dockerfile_dir=$(cd $(dirname $0); pwd)

docker build -t nn_with_c:${USER} ${dockerfile_dir}
