.PHONY: release debug test sample clean

BUILD_DIR=./build

release:
	@cmake -DCMAKE_BUILD_TYPE=Release -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR)

debug:
	@cmake -DCMAKE_BUILD_TYPE=Debug -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR)

sample:
	@cmake -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR) --target sample

# Run all test cases in default
CASE=all

test:
	@./docker/docker_run.sh "ceedling test:$(CASE)"


clean:
	@cmake --build $(BUILD_DIR) --target clean
