import os

from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    declare_spawn_x_val = DeclareLaunchArgument('spawn_x_val', default_value='0.0', description='Spawn X Value')
    declare_spawn_y_val = DeclareLaunchArgument('spawn_y_val', default_value='0.0', description='Spawn Y Value')
    declare_spawn_yaw_val = DeclareLaunchArgument('spawn_yaw_val', default_value='0.0', description='Spawn Yaw Value')

    # Package directories
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    description_path = get_package_share_directory("pdd_description")
    default_model_path = os.path.join(description_path, "xacro/robot.xacro")

    # Gazebo server
    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'extra_gazebo_args': '--ros-args'}.items()
    )

    # Gazebo client
    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
    )

    # Robot description
    robot_description_config = Command(['xacro ', default_model_path])
    params = {'robot_description': ParameterValue(robot_description_config, value_type=str)}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'pdd',
            '-x', LaunchConfiguration('spawn_x_val'),
            '-y', LaunchConfiguration('spawn_y_val'),
            '-z', '0.8',
            '-Y', LaunchConfiguration('spawn_yaw_val')
        ],
        output='screen'
    )

    # Controllers
    load_controller_manager = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            'joint_state_broadcaster',
            'leg_l1_joint_controller',
            'leg_l2_joint_controller',
            'leg_l3_joint_controller',
            'leg_l4_joint_controller',
            'leg_l5_joint_controller',
            'leg_r1_joint_controller',
            'leg_r2_joint_controller',
            'leg_r3_joint_controller',
            'leg_r4_joint_controller',
            'leg_r5_joint_controller',
        ],
        output='screen',
    )

    # Delay controller loading until robot is spawned
    controller_manager_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[load_controller_manager],
        )
    )

    # RViz2
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(description_path, 'rviz/default.rviz')]
    )

    # Joystick
    joy = Node(
        package='joy_linux',
        executable='joy_linux_node',
        name='joystick',
        output='screen'
    )

    return LaunchDescription([
        declare_spawn_x_val,
        declare_spawn_y_val,
        declare_spawn_yaw_val,
        start_gazebo_server,
        start_gazebo_client,
        spawn_entity,
        robot_state_publisher,
        controller_manager_spawner,
        rviz2_node,
        joy,
    ]) 