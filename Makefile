SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:

DOCKER_CONTAINER := moveit
DOCKER_CONTAINER_VGL := ${DOCKER_CONTAINER}_vgl
DOCKER_IMAGE := visincept-cn-shanghai.cr.volces.com/grasp/sif:v7.0
IP := 192.168.211.31
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
		${DOCKER_IMAGE} \
		bash -c "/bin/bash"

docker_vgl:
	if [ -z "$$DISPLAY" ]; then
		echo "DISPLAY is empty. Please run in a GUI/X11 session." >&2
		exit 1
	fi
	if [[ "$$DISPLAY" =~ ^:[0-9]+([.][0-9]+)?$$ ]]; then
		if [ ! -S /tmp/.X11-unix/X$${DISPLAY%%.*} ]; then
			echo "X11 socket for DISPLAY=$$DISPLAY not found under /tmp/.X11-unix." >&2
			echo "Check your DISPLAY value or X server status first." >&2
			exit 1
		fi
	else
		echo "DISPLAY=$$DISPLAY is not a local Unix socket display; skip socket file check."
	fi
	docker run -it --rm \
		--name ${DOCKER_CONTAINER_VGL} \
		--shm-size=8g \
		--network host \
		--ipc host \
		--gpus all \
		-e DISPLAY=$$DISPLAY \
		-e XAUTHORITY=/root/.Xauthority \
		-e VGL_CLIENT=${IP}:0 \
		-e QT_X11_NO_MITSHM=1 \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $$HOME/.Xauthority:/root/.Xauthority:ro \
		-v /data1/shock/workspace/service/..:/workspace \
		-v /data1:/data1 \
		-v /comp_robot/shock/hf_cache:/root/.cache \
		-v /comp_robot/shock/hf_cache:/comp_robot/shock/hf_cache \
		-v /data1/comp_robot:/comp_robot \
		-v /data1/vePFS:/vePFS \
		-w /workspace/moveit-service \
		-e ROS_DOMAIN_ID=10 \
		${DOCKER_IMAGE} \
		bash -c "/bin/bash"

exec:
	docker exec -it  -e DISPLAY=:1 -w /workspace/moveit-service ${DOCKER_CONTAINER} bash

exec_vgl:
	docker exec -it \
		-e DISPLAY=$DISPLAY \
		-e XAUTHORITY=/root/.Xauthority \
		-e QT_X11_NO_MITSHM=1 \
		-w /workspace/moveit-service \
		${DOCKER_CONTAINER_VGL} bash


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

run-vis-vgl:
	source set-env.sh
	echo "DISPLAY=$${DISPLAY}"
	vglrun -d $${VGL_DISPLAY:-egl} ros2 launch xarm7_moveit_config xarm7_sim_planning.launch.py

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

.PHONY: set-env docker docker_vgl exec exec_vgl build run run-vis run-vis-vgl test status clean
