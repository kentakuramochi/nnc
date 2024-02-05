.PHONY: test

test:
	@./docker/docker_run.sh "ceedling test"

