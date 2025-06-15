# my_robot_launch.launch.py 예시

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # 노드 1: IMU 데이터 노드
        Node(
            package='find_maze',
            executable='imu',
            name='imu_data_node',
            #output='screen',
        ),

        # 노드 2: IMU 데이터 노드 (예시)
        Node(
            package='find_maze',
            executable='teleop',
            name='teleop_control_node',
            #output='screen',
        ),

        # 노드 3: right follower node
        Node(
            package='find_maze',
            executable='back',  # setup.py의 entry_points에 정의된 이름
            name='cell_navigator_action_server_node', # 런치 파일 내에서 이 노드를 식별할 이름
            output='screen', # 노드의 로그를 터미널에 출력
            #parameters=[
            #    # 여기에 노드에 전달할 매개변수 (파라미터)를 정의할 수 있습니다.
            #    # 예: {'my_param': 1.0}
            #]
        ),

    ])
