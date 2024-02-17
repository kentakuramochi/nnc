.PHONY: release test clean

release:
	@cmake -DCMAKE_BUILD_TYPE=Release -B ./build . && cmake --build ./build

debug:
	@cmake -DCMAKE_BUILD_TYPE=Debug -B ./build . && cmake --build ./build

test:
	@./docker/docker_run.sh "ceedling test"

clean:
	@./docker/docker_run.sh "ceedling clean"
