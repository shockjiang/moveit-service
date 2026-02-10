#!/usr/bin/env python3
# xarm7 simulation launch file - Load sensors from YAML file
# Correctly loads octomap sensor configuration from external YAML

import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler, EmitEvent, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import xacro
import yaml


def load_yaml_file(package_name, file_path):
    """Load YAML file from package"""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError as e:
        print(f"Error loading {absolute_file_path}: {e}")
        return None


def launch_setup(context, *_args, **_kwargs):
    
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
            'ros2_control_plugin': 'mock_components/GenericSystem',
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
    # KINEMATICS
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
    # JOINT LIMITS
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
    # PLANNING PIPELINE (OMPL)
    # ==========================================
    ompl_planning_full = {
        'planning_pipelines': ['ompl'],
        'ompl': {
            'planning_plugins': ['ompl_interface/OMPLPlanner'],
            'request_adapters': [
                'default_planning_request_adapters/ResolveConstraintFrames',
                'default_planning_request_adapters/ValidateWorkspaceBounds',
                'default_planning_request_adapters/CheckStartStateBounds',
                'default_planning_request_adapters/CheckStartStateCollision'
            ],
            'response_adapters': [
                'default_planning_response_adapters/AddTimeOptimalParameterization',
                'default_planning_response_adapters/ValidateSolution',
                'default_planning_response_adapters/DisplayMotionPath'
            ],
            'start_state_max_bounds_error': 0.1,
            'jiggle_fraction': 0.05,
            'projection_evaluator': 'joints(joint1,joint2)',
            'longest_valid_segment_fraction': 0.005,
            'planner_configs': {
                'RRTConnect': {
                    'type': 'geometric::RRTConnect',
                    'range': 0.0
                },
                'RRT': {
                    'type': 'geometric::RRT',
                    'range': 0.0,
                    'goal_bias': 0.05
                },
                'RRTstar': {
                    'type': 'geometric::RRTstar',
                    'range': 0.0,
                    'goal_bias': 0.05,
                    'delay_collision_checking': 1
                },
                'TRRT': {
                    'type': 'geometric::TRRT',
                    'range': 0.0,
                    'goal_bias': 0.05
                },
                'PRM': {
                    'type': 'geometric::PRM',
                    'max_nearest_neighbors': 10
                },
                'PRMstar': {
                    'type': 'geometric::PRMstar'
                },
                'SBL': {
                    'type': 'geometric::SBL',
                    'range': 0.0
                },
                'EST': {
                    'type': 'geometric::EST',
                    'range': 0.0,
                    'goal_bias': 0.05
                },
                'LBKPIECE': {
                    'type': 'geometric::LBKPIECE',
                    'range': 0.0,
                    'border_fraction': 0.9,
                    'min_valid_path_fraction': 0.5
                },
                'BKPIECE': {
                    'type': 'geometric::BKPIECE',
                    'range': 0.0,
                    'border_fraction': 0.9,
                    'failed_expansion_score_factor': 0.5,
                    'min_valid_path_fraction': 0.5
                },
                'KPIECE': {
                    'type': 'geometric::KPIECE',
                    'range': 0.0,
                    'goal_bias': 0.05,
                    'border_fraction': 0.9,
                    'failed_expansion_score_factor': 0.5,
                    'min_valid_path_fraction': 0.5
                }
            },
            'xarm7': {
                'default_planner_config': 'RRTConnect',
                'planner_configs': ['SBL', 'EST', 'LBKPIECE', 'BKPIECE', 'KPIECE', 'RRT', 'RRTConnect', 'RRTstar', 'TRRT', 'PRM', 'PRMstar']
            },
            'xarm_gripper': {
                'default_planner_config': 'RRTConnect',
                'planner_configs': ['RRTConnect', 'RRT']
            }
        }
    }

    # ==========================================
    # TRAJECTORY EXECUTION
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
            'allowed_start_tolerance': 0.5,
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
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
        'publish_planning_scene_hz': 10.0,
        'publish_geometry_updates_hz': 10.0,
        'publish_state_updates_hz': 10.0,
        'publish_transforms_updates_hz': 10.0,
        'monitored_planning_scene': '/monitored_planning_scene',
    }

    # ==========================================
    # LOAD SENSOR CONFIGURATION FROM YAML FILE
    # ==========================================
    # Try to load from your config file path
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
    # MOVEIT CONFIGURATION
    # ==========================================
    moveit_config = {
        'collision_detector': 'FCL',
    }

    # ==========================================
    # ROS2 CONTROL PARAMETERS
    # ==========================================
    ros2_controllers_file = os.path.join(
        get_package_share_directory('xarm7_moveit_config'),
        'config',
        'ros2_controllers.yaml'
    )

    # ==========================================
    # NODES
    # ==========================================
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # Static TF
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_broadcaster',
        output='screen',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '0', '--pitch', '0', '--yaw', '0', '--frame-id', 'world', '--child-frame-id', 'link_base'],
    )

    # ROS2 Control
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[robot_description, ros2_controllers_file],
    )

    # Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    delayed_joint_state_broadcaster = TimerAction(
        period=2.0,
        actions=[joint_state_broadcaster_spawner],
    )

    # Arm Controller
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['xarm7_traj_controller', '--controller-manager', '/controller_manager'],
    )
    delayed_arm_controller = TimerAction(
        period=3.0,
        actions=[arm_controller_spawner],
    )

    # Gripper Controller
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['xarm_gripper_traj_controller', '--controller-manager', '/controller_manager'],
    )
    delayed_gripper_controller = TimerAction(
        period=3.5,
        actions=[gripper_controller_spawner],
    )

    # Move Group
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            joint_limits_yaml,
            ompl_planning_full,
            trajectory_execution,
            planning_scene_monitor,
            sensors_3d_yaml,
            moveit_config,
        ],
    )

    # RViz
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
            ompl_planning_full,
            joint_limits_yaml,
            planning_scene_monitor,
        ],
    )

    # Shutdown on RViz exit
    rviz_exit_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=rviz_node,
            on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    return [
        rviz_exit_handler,
        robot_state_publisher,
        static_tf_node,
        ros2_control_node,
        delayed_joint_state_broadcaster,
        delayed_arm_controller,
        delayed_gripper_controller,
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])