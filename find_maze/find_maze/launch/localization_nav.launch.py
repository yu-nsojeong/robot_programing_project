from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument # 런치 인수 선언을 위해 추가
from launch.substitutions import LaunchConfiguration # 런치 인수 값을 사용하기 위해 추가

def generate_launch_description():
    # 1. 런치 인수 선언
    #    'target_row'와 'target_col'이라는 런치 인수를 정의합니다.
    #    사용자가 값을 지정하지 않으면 '0'과 '4'를 기본값으로 사용합니다.
    target_row_arg = DeclareLaunchArgument(
        'target_row',
        default_value='0',
        description='The target cell row number for navigation.'
    )
    target_col_arg = DeclareLaunchArgument(
        'target_col',
        default_value='4',
        description='The target cell column number for navigation.'
    )

    # 2. 런치 인수 값 가져오기 (LaunchConfiguration 객체 생성)
    #    선언된 런치 인수의 현재 값을 가져와 노드 파라미터로 전달할 수 있도록 준비합니다.
    target_row_lc = LaunchConfiguration('target_row')
    target_col_lc = LaunchConfiguration('target_col')

    return LaunchDescription([
        # 선언된 런치 인수들을 LaunchDescription에 추가해야 합니다.
        target_row_arg,
        target_col_arg,

        # 노드 1: IMU 데이터 노드
        Node(
            package='find_maze',
            executable='imu',
            name='imu_data_node',
            # output='screen', # 기본값 'log' (터미널에 직접 출력 안함)
        ),

        # 노드 2: teleop 노드 (이름은 고유하게 잘 변경하셨습니다!)
        Node(
            package='find_maze',
            executable='teleop',
            name='teleop_control_node',
            # output='screen', # 기본값 'log'
        ),

        # 노드 3: Action Server ('as')
        Node(
            package='find_maze',
            executable='as',
            name='cell_navigator_action_server_node',
            # output='screen', # Action Server의 로그를 터미널에서 보려면 활성화
            # parameters=[ # 서버에 전달할 파라미터가 있다면 여기에 추가
            #     {'yaml_path': '/path/to/your/map.yaml'},
            #     {'pgm_path': '/path/to/your/map.pgm'},
            # ]
        ),

        # 노드 4: Action Client ('ac')
        Node(
            package='find_maze',
            executable='ac',
            name='cell_navigator_action_client_node',
            output='screen', # Action Client의 로그는 터미널에 출력
            parameters=[
                # 런치 인수를 노드 파라미터로 전달합니다. 이 부분이 핵심입니다!
                {'target_row': target_row_lc},
                {'target_col': target_col_lc}
            ]
        ),
    ])
