# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
This project aims to implement a xarm7 + bio-gripper path planning with standard ros2 and moveit2. We can create urdf/srdf and MovingGroup from resources under ../libs/xarm_ros2/xarm_description, but we should never rely on package or code of xarm_ros2.

Our code should run under the docker container named "sif", which has installed standard jazzy ros2 and moveit packages. Running command: docker exec  sif bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace/moveit; /usr/bin/python3 test_moveit.py" can test it

We can break our the above goal to multiple steps:
- `launch_xarm_with_gripper.py` - create urdf for xarm7 and bio-gripper
- `launch_xarm_simple.py` - create movinggroup for xarm7 and run this moving group as a ros node
- `test_xarm7_planning.py` - test xarm7 path planing

there are some other resources you can ref to:
- `moveit2_tutorials/` - Official MoveIt 2 tutorials (from upstream repository)
- `../libs/xarm_ros2/` - Official xarm resources including integration with moveit. You should never repy on or install these package, but you can ref to this code for your implementation.


**MoveIt** is the standard motion planning framework for ROS 2. It provides:
- Motion planning (OMPL-based)
- Collision checking with planning scene
- IK/FK kinematics solvers
- Cartesian path planning
- Trajectory execution

## Development Environment

**ROS Distribution**: ROS 2 Jazzy
**Container**: Docker container named `sif`
**Python**: `/usr/bin/python3` (system Python, not venv)

### Source Setup
All ROS commands require sourcing the environment first:
```bash
source /opt/ros/jazzy/setup.bash
```

## Common Commands

### Starting MoveIt
```bash
# Start MoveIt demo with Panda robot (in background)
make start

# Check if MoveIt is running
make status

# Stop all MoveIt nodes
make stop
```

### Running Tests
```bash
# Run tests (requires MoveIt already running)
make test

# Auto-start MoveIt if needed, then run tests
make debug

# try path planning
make plan-xarm7
```

### Manual Execution
```bash
# Enter Docker container
docker exec -it sif bash

# Source ROS environment
source /opt/ros/jazzy/setup.bash

```

## Architecture

### MoveIt Data Flow

```
Target Pose → IK Solver → Joint Goals → Motion Planner → Trajectory → Controller → Robot
                                ↑
                         Planning Scene
                      (collision objects)
```

### Key MoveIt Interfaces

**Services** (synchronous requests):
- `/get_planning_scene` - Get current planning scene state
- `/apply_planning_scene` - Add/remove collision objects
- `/compute_ik` - Inverse kinematics solver
- `/compute_fk` - Forward kinematics solver
- `/compute_cartesian_path` - Straight-line Cartesian motion

**Actions** (asynchronous, long-running):
- `/move_action` - Motion planning with MoveGroup
- `/execute_trajectory` - Execute a planned trajectory

**Topics**:
- `/camera/depth/color/points` - RGB-D point cloud (RealSense)
- `/joint_states` - Current robot joint positions

### Planning Scene Management

The planning scene is the **world model** that MoveIt uses for collision checking:
- **World objects**: Static obstacles (tables, walls, etc.)
- **Robot state**: Current joint positions and attached objects
- **Collision objects**: Primitives (box, cylinder, sphere) or meshes

**Critical**: Always update the planning scene before planning. Motion planning without collision objects ignores all obstacles.

### Motion Planning Modes

**1. Joint Space Planning**
Direct joint-to-joint motion. Fastest but path is unpredictable in Cartesian space.
```python
plan_to_joint_positions(joint_positions)
```

**2. Pose Planning (IK + Joint Space)**
Converts Cartesian target to joints via IK, then plans in joint space.
```python
plan_to_pose(target_pose)  # Uses compute_ik() internally
```

**3. Cartesian Path Planning**
Straight-line motion through waypoints. Slower but predictable path.
```python
plan_cartesian_path(waypoints, max_step=0.01)
```

### Grasp Sequence Pattern

Standard pick-and-place motion sequence:
```
1. Move to pre-grasp pose (above object) - Joint planning
2. Approach (straight down to grasp) - Cartesian path
3. Close gripper - Hardware-specific
4. Retreat (lift up) - Cartesian path
5. Move to place location - Joint planning
6. Open gripper - Hardware-specific
```

**Why this pattern?**
- Joint planning for free-space motion (fast, collision-aware)
- Cartesian paths for controlled approach/retreat (maintains orientation, predictable)

## Robot Configuration

### For xArm Robot

**Planning group**: `xarm7`
**Base frame**: `link_base`
**Joint names**: `joint1` through `joint7` (xArm7)

**IMPORTANT**: `test_xarm_grasp.py` has hardcoded Panda joint names (line 352-356) despite claiming xArm support. This is a bug that will cause failures on real xArm hardware. The demo function (line 580) correctly uses "panda_arm" group, but production code needs dynamic joint name resolution.

## Code Patterns

### MoveIt Node Structure
```python
class MyPlanner(Node):
    def __init__(self):
        super().__init__('my_planner')

        # Action clients (async, long-running)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')

        # Service clients (sync, fast)
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # Wait for services
        self.move_client.wait_for_server(timeout_sec=10.0)
        self.scene_client.wait_for_service(timeout_sec=10.0)
```

### Planning Scene Updates
```python
# Create collision object
co = CollisionObject()
co.id = "obstacle"
co.operation = CollisionObject.ADD
co.primitives = [box]
co.primitive_poses = [pose]

# Apply to scene
scene = PlanningScene()
scene.is_diff = True  # Incremental update
scene.world.collision_objects = [co]

request = ApplyPlanningScene.Request()
request.scene = scene
future = self.scene_client.call_async(request)
```

### Motion Planning
```python
# Create goal
goal = MoveGroup.Goal()
goal.request.group_name = "panda_arm"
goal.request.allowed_planning_time = 5.0
goal.planning_options.plan_only = True  # Don't execute

# Set constraints (joint or pose)
goal.request.goal_constraints.append(constraints)

# Send to planner
future = self.move_client.send_goal_async(goal)
goal_handle = future.result()
result_future = goal_handle.get_result_async()
result = result_future.result().result

if result.error_code.val == 1:  # SUCCESS
    trajectory = result.planned_trajectory
```

## Test Files

### test_moveit.py
Basic MoveIt functionality verification suite. Tests:
1. Planning scene access
2. Forward kinematics (FK)
3. Inverse kinematics (IK)
4. Motion planning with OMPL
5. Cartesian path planning

**Usage**: Diagnostics and regression testing after MoveIt setup changes.

### test_xarm_grasp.py
Grasp planning wrapper with collision avoidance. Demonstrates:
- Adding collision objects (tables, obstacles)
- Planning collision-free paths
- Grasp sequence execution (pre-grasp → approach → retreat)
- Attaching/detaching objects to gripper

**Usage**: Template for pick-and-place applications. Contains reusable helper methods for collision scene management.

## Common Issues

### "Service not available"
MoveIt is not running. Use `make start` or check with `make status`.

### "Planning failed"
- Check if target pose is reachable (IK must have solution)
- Verify planning scene is updated with current obstacles
- Increase `allowed_planning_time` or `num_planning_attempts`
- Check if robot is in collision with obstacles

### "IK failed"
Target pose is outside workspace or violates joint limits. Try:
- Moving target closer to robot base
- Adjusting orientation
- Using a different approach angle

### Cartesian path incomplete (fraction < 1.0)
Cartesian planner hit an obstacle or joint limit. Solutions:
- Reduce waypoint distances
- Adjust `max_step` parameter
- Remove blocking collision objects
- Use joint planning for free-space motion instead

## File Locations

**Production code**: Root directory (`test_moveit.py`, `test_xarm_grasp.py`)
**MoveIt tutorials**: `moveit2_tutorials/doc/`
**Launch files**: `moveit2_tutorials/doc/*/launch/*.launch.py`
**Config files**: `moveit2_tutorials/doc/*/config/*.yaml`

## Dependencies

Install required ROS packages:
```bash
make install
```

This installs:
- `ros-jazzy-joint-state-broadcaster` - Joint state controller
- `ros-jazzy-joint-trajectory-controller` - Trajectory execution
- `ros-jazzy-ros2controlcli` - Controller management CLI

## References

- [MoveIt 2 Documentation](https://moveit.picknik.ai/)
- [MoveIt 2 Tutorials Repository](https://github.com/moveit/moveit2_tutorials)
- [ROS 2 Jazzy Documentation](https://docs.ros.org/en/jazzy/)
