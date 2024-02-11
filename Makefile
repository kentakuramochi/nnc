.PHONY: release test clean

release:
	@./docker/docker_run.sh "ceedling release"

test:
	@./docker/docker_run.sh "ceedling test"

clean:
	@./docker/docker_run.sh "ceedling clean"
