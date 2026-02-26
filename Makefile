SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:

DOCKER_CONTAINER := moveit
#sudo apt install -y ros-jazzy-octomap-server ros-jazzy-octomap-ros ros-jazzy-octomap-rviz-plugins
set-env:
	source set-env.sh


docker:
	docker run -it --rm \
		--name ${DOCKER_CONTAINER} \
		--shm-size=8g \
		-v /data1/shock/workspace/service/..:/workspace \
		-v /data1:/data1 \
		-v /comp_robot/shock/hf_cache:/root/.cache \
		-v /comp_robot/shock/hf_cache:/comp_robot/shock/hf_cache \
		-v /data1/comp_robot:/comp_robot \
		-v /data1/vePFS:/vePFS \
		-w /workspace/moveit-service \
		-e ROS_DOMAIN_ID=10 \
		visincept-cn-shanghai.cr.volces.com/grasp/sif:v7.0 \
		bash -c "/bin/bash"

exec:
	docker exec -it  -e DISPLAY=:1 -w /workspace/moveit-service ${DOCKER_CONTAINER} bash


debug:
	source set-env.sh
	/usr/bin/python3 pointcloud_to_moveit_convexhull.py
	#/usr/bin/python3 tests/create_scene_with_pc.py


build:
	source set-env.sh
	colcon build

run: # must be run with rviz
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py use_rviz:=false

octo-server:
	source set-env.sh
	ros2 run octomap_server octomap_server_node --ros-args \
		-p frame_id:=world \
		-p resolution:=0.005 \
		-r cloud_in:=/camera/depth/color/points

service:
	source set-env.sh
	/usr/bin/python3 moveit_service.py

client:
	source set-env.sh
	/usr/bin/python3 grasp_client.py --target 3

# 	xvfb-run ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py use_rviz:=false 2 > log/moveit.log & tail -f log/moveit.log

run-vis:
	source set-env.sh
	export DISPLAY=:1
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
	pkill -f /opt/ros/
	pkill -9 Xvfb
	

start-all:
	source set-env.sh
	ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py

.PHONY: set-env exec build run test status clean
