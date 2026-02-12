SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c


exec:
	docker exec -it  -e DISPLAY=:1 -w /workspace/moveit-service sif bash

build:
	source /opt/ros/jazzy/setup.bash && colcon build

run:
	source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py

clean:
	rm -fr install log build

