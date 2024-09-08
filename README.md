# nn-with-c

Simple implementation of the neural network with C language, w/o third party libraries

## Directories and files

```
nn-with-c/
  |- docker/: Docker configs
  |- include/: Library headers
  |- sample/: Sample programs
  |- src/: Library sources
  |- test/: Unit tests
  |- CMakeLists.txt: Top-level CMake config
  |- LICENSE
  |- Makefile: Top-level task runner
  |- project.yml: ceedling configs
  `- README.md
```

## Prerequisites

- gcc (C99)
- GNU Make
- CMake (> 3.16.0)

### Prerequisites for unit tests

- [ThrowTheSwitch/Ceedling](https://github.com/ThrowTheSwitch/Ceedling): Ruby-based unit testing and build system for C project
    - [ThrowTheSwitch/Unity](https://github.com/ThrowTheSwitch/Unity): Unit test framework for C
    - [ThrowTheSwitch/CMock](https://github.com/ThrowTheSwitch/CMock): Mock/stub generator for C

- Docker: build enviromnent

    Build a Docker image named `nn_with_c` with ceedling system by:

    ```sh
    $ ./docker/docker_build.sh
    ```

    Run a specified command by using the Docker image by:

    ```sh
    $ ./docker/docker_run.sh <COMMAND>
    ```

## Build

### Build a library

Build a library `libnn` by:

```sh
$ make
$ make release # Release build (default)
$ make debug  # Debug build
```

### Build samples

Build sample programs in `sample` by:

```sh
$ make sample
```

## Test

Run unit tests in `test` by:

```sh
$ make test
```

It requires the ceedling container.

## Clean

```sh
$ make clean
```

## License

MIT License
