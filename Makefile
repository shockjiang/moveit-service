SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:

set-env:
	source set-env.sh

exec:
	docker exec -it  -e DISPLAY=:1 -w /workspace/moveit-service sif bash

build:
	source set-env.sh
	colcon build

run:
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py


test:
	source set-env.sh
	/usr/bin/python3 tests/move_arm_simple.py


status:
	source set-env.sh
	ros2 node list


clean:
	rm -fr install log build

.PHONY: set-env exec build run test status clean
