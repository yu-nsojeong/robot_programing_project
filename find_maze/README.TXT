

패키지 빌드는 잘 할거라 밑고,,,
필요 패키지

dynamixlSDK
find_maze
find_maze_interfaces
turtlebot3
turtlebot3_msg
turtlebot3_simulations



#런치로 localization_nave와 right_wall_follpwer실행 가능

#시뮬로 하려면 먼저
ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py

#5*5 행렬임.(시뮬 맵 왼쪽 위 기준 인덱스 0-4,0-4)
#런치 실행 할 때 target 값을 바꿔 줘서 목표 지점 선택 가능
ros2 launch find_maze localization_nav.launch.py target_row:=4 target_col:=4
-> 아무데나 시작해도 알아서 로컬라이징 해서 목표 도착 가능함


