#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Standalone xarm7 simulation launch file - MINIMAL DEPENDENCIES VERSION
# Only requires URDF/SRDF files, all configurations are inlined
# Perfect for easy migration

import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import xacro
import yaml
import tempfile


def load_file(package_name, file_path):
    """Load file content"""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None


def load_yaml_file(package_name, file_path):
    """Load YAML file"""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def launch_setup(context, *args, **kwargs):
    
    # ==========================================
    # ROBOT DESCRIPTION (URDF)
    # ==========================================
    xacro_file = os.path.join(
        get_package_share_directory('xarm7_description'),
        'urdf', 'xarm_device.urdf.xacro'
    )
    
    robot_description_config = xacro.process_file(
        xacro_file,
        mappings={
            'dof': '7',
            'robot_type': 'xarm',
            'prefix': '',
            'hw_ns': 'xarm',
            'limited': 'true',
            'effort_control': 'false',
            'velocity_control': 'false',
            'add_gripper': 'true',
            'add_vacuum_gripper': 'false',
            'add_bio_gripper': 'false',
            'add_realsense_d435i': 'false',
            'add_other_geometry': 'false',
            'ros2_control_plugin': 'uf_robot_hardware/UFRobotFakeSystemHardware',
            'attach_to': 'world',
            'attach_xyz': '0 0 0',
            'attach_rpy': '0 0 0',
        }
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    # ==========================================
    # ROBOT SEMANTIC DESCRIPTION (SRDF)
    # ==========================================
    srdf_file = os.path.join(
        get_package_share_directory('xarm7_moveit_config'),
        'srdf', 'xarm.srdf.xacro'
    )
    
    robot_description_semantic_config = xacro.process_file(
        srdf_file,
        mappings={
            'prefix': '',
            'dof': '7',
            'robot_type': 'xarm',
            'add_gripper': 'true',
            'add_vacuum_gripper': 'false',
            'add_bio_gripper': 'false',
            'add_other_geometry': 'false',
        }
    )
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_config.toxml()
    }

    # ==========================================
    # KINEMATICS (Inline Configuration)
    # ==========================================
    kinematics_yaml = {
        'robot_description_kinematics': {
            'xarm7': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_search_resolution': 0.005,
                'kinematics_solver_timeout': 0.005,
                'kinematics_solver_attempts': 3,
            }
        }
    }

    # ==========================================
    # JOINT LIMITS (Inline Configuration)
    # ==========================================
    joint_limits_yaml = {
        'robot_description_planning': {
            'joint_limits': {
                'joint1': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint2': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint3': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint4': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint5': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint6': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'joint7': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': True, 'max_acceleration': 2.0},
                'drive_joint': {'has_velocity_limits': True, 'max_velocity': 2.0, 'has_acceleration_limits': True, 'max_acceleration': 5.0},
            }
        }
    }

    # ==========================================
    # PLANNING PIPELINE (Inline Configuration)
    # ==========================================
    ompl_planning_yaml = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
            'xarm7': {
                'default_planner_config': 'RRTConnect',
                'planner_configs': ['SBL', 'EST', 'LBKPIECE', 'BKPIECE', 'KPIECE', 'RRT', 'RRTConnect', 'RRTstar', 'TRRT', 'PRM', 'PRMstar'],
            },
            'xarm_gripper': {
                'default_planner_config': 'RRTConnect',
                'planner_configs': ['RRTConnect', 'RRT'],
            }
        }
    }

    # ==========================================
    # TRAJECTORY EXECUTION (Inline Configuration)
    # ==========================================
    moveit_simple_controllers_yaml = {
        'controller_names': ['xarm7_traj_controller', 'xarm_gripper_traj_controller'],
        'xarm7_traj_controller': {
            'action_ns': 'follow_joint_trajectory',
            'type': 'FollowJointTrajectory',
            'default': True,
            'joints': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
        },
        'xarm_gripper_traj_controller': {
            'action_ns': 'follow_joint_trajectory',
            'type': 'FollowJointTrajectory',
            'default': True,
            'joints': ['drive_joint'],
        }
    }

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'trajectory_execution': {
            'allowed_execution_duration_scaling': 1.2,
            'allowed_goal_duration_margin': 0.5,
            'allowed_start_tolerance': 0.01,
        }
    }

    # ==========================================
    # PLANNING SCENE MONITOR
    # ==========================================
    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # ==========================================
    # OCTOMAP CONFIGURATION (Inline)
    # ==========================================
    sensors_3d_yaml = {
        'sensors': ['ros'],
        'octomap_frame': 'world',
        'octomap_resolution': 0.02,
        'ros': {
            'sensor_plugin': 'occupancy_map_monitor/PointCloudOctomapUpdater',
            'point_cloud_topic': '/camera/depth/color/points',
            'max_range': 2.0,
            'point_subsample': 1,
            'padding_offset': 0.1,
            'padding_scale': 1.0,
            'max_update_rate': 5.0,
            'filtered_cloud_topic': '/filtered_cloud',
        }
    }

    # ==========================================
    # ROS2 CONTROL PARAMETERS (Inline)
    # ==========================================
    # Create temporary controller config file
    ros2_controllers_yaml = {
        'controller_manager': {
            'ros__parameters': {
                'update_rate': 100,
                'joint_state_broadcaster': {
                    'type': 'joint_state_broadcaster/JointStateBroadcaster',
                },
                'xarm7_traj_controller': {
                    'type': 'joint_trajectory_controller/JointTrajectoryController',
                },
                'xarm_gripper_traj_controller': {
                    'type': 'joint_trajectory_controller/JointTrajectoryController',
                },
            }
        },
        'xarm7_traj_controller': {
            'ros__parameters': {
                'joints': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
                'command_interfaces': ['position'],
                'state_interfaces': ['position', 'velocity'],
                'state_publish_rate': 100.0,
                'action_monitor_rate': 20.0,
                'allow_partial_joints_goal': False,
            }
        },
        'xarm_gripper_traj_controller': {
            'ros__parameters': {
                'joints': ['drive_joint'],
                'command_interfaces': ['position'],
                'state_interfaces': ['position', 'velocity'],
                'state_publish_rate': 100.0,
                'action_monitor_rate': 20.0,
            }
        },
    }
    
    # Write to temporary file
    ros2_control_params_file = tempfile.NamedTemporaryFile(mode='w', prefix='xarm7_controllers_', suffix='.yaml', delete=False)
    yaml.dump(ros2_controllers_yaml, ros2_control_params_file, default_flow_style=False)
    ros2_control_params_file.close()

    # ==========================================
    # NODE 1: Robot State Publisher
    # ==========================================
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # ==========================================
    # NODE 2: ROS2 Control Node
    # ==========================================
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[robot_description, ros2_control_params_file.name],
    )

    # ==========================================
    # NODE 3: Joint State Broadcaster
    # ==========================================
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    # ==========================================
    # NODE 4: Arm Trajectory Controller
    # ==========================================
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['xarm7_traj_controller', '--controller-manager', '/controller_manager'],
    )

    # ==========================================
    # NODE 5: Gripper Trajectory Controller
    # ==========================================
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['xarm_gripper_traj_controller', '--controller-manager', '/controller_manager'],
    )

    # ==========================================
    # NODE 6: MoveIt Move Group
    # ==========================================
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            joint_limits_yaml,
            ompl_planning_yaml,
            trajectory_execution,
            planning_scene_monitor,
            sensors_3d_yaml,
        ],
    )

    # ==========================================
    # NODE 7: RViz2
    # ==========================================
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('xarm7_moveit_config'),
        'rviz',
        'moveit.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_yaml,
            joint_limits_yaml,
        ],
    )

    # ==========================================
    # NODE 8: Static TF Publisher (world -> link_base)
    # ==========================================
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'link_base'],
    )

    # ==========================================
    # Event Handler: Shutdown on RViz Exit
    # ==========================================
    rviz_exit_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=rviz_node,
            on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    return [
        rviz_exit_handler,
        robot_state_publisher,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
        static_tf_node,
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
