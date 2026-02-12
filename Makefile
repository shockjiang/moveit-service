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

run: # must be run with rviz
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py use_rviz:=false

# 	xvfb-run ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py use_rviz:=false 2 > log/moveit.log & tail -f log/moveit.log

run-vis:
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py 

test:
	source set-env.sh
	/usr/bin/python3 tests/move_arm_simple.py


status:
	source set-env.sh
	ros2 node list
	ps aux|grep ros

clean:
	rm -fr install log build


stop-all:
	pkill -9 -f ros2
	pkill -9 -f ros
	pkill -9 -f _ros2_daemon
	pkill -9 -f /opt/ros/
	pkill -9 -f /move_group
	pkill -9 Xvfb
	

start-all:
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py

.PHONY: set-env exec build run test status clean
