.PHONY: release test clean

release:
	@cmake -DCMAKE_BUILD_TYPE=Release -B ./build . && cmake --build ./build

debug:
	@cmake -DCMAKE_BUILD_TYPE=Debug -B ./build . && cmake --build ./build

# Run all test cases in default
CASE=all

test:
	@./docker/docker_run.sh "ceedling test:$(CASE)"

clean:
	@cmake --build ./build --target clean
