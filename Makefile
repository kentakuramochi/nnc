.PHONY: release test example clean

BUILD_DIR=./build

release:
	@cmake -DCMAKE_BUILD_TYPE=Release -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR)

debug:
	@cmake -DCMAKE_BUILD_TYPE=Debug -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR)

example:
	@cmake -B $(BUILD_DIR) . && cmake --build $(BUILD_DIR) --target example

# Run all test cases in default
# CASE=all

test:
#	@./docker/docker_run.sh "ceedling test:$(CASE)"
	@./docker/docker_run.sh "ceedling test:test_nn_net test:test_nn_layer test:test_nn_trainer test:test_nn_loss \
    test:test_fc_layer test:test_sigmoid_layer"


clean:
	@cmake --build $(BUILD_DIR) --target clean
