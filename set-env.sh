# Source ROS environment
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "Error: ROS setup file not found at /opt/ros/jazzy/setup.bash" >&2
    return 1
fi

# Source workspace overlay
if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "Warning: Workspace setup file not found at install/setup.bash" >&2
    echo "Run 'colcon build' first to create the workspace overlay" >&2
fi

alias python=/usr/bin/python3
alias python3=/usr/bin/python3
# Only prepend /usr/bin to PATH if it's not already first
if [[ ":$PATH:" != ":/usr/bin:"* ]]; then
    export PATH=/usr/bin:$PATH
fi