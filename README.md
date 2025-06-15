# 🐢 TurtleBot3 미로 탈출 프로젝트

TurtleBot3와 ROS 2를 이용해 미로를 탐색하고, 탈출하며, 목표 지점까지 스스로 도달하는 프로젝트입니다.  
SLAM, Localization, A* 경로 계획 등을 통해 자율적으로 미로 환경을 이해하고 주행할 수 있습니다.  
시뮬레이션 환경에서도 동일한 기능을 테스트할 수 있습니다.

![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange?logo=ubuntu)
![ROS2](https://img.shields.io/badge/ROS-2%20Humble-blue?logo=ros)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 주요 기능

1. **오른쪽 우선 백트래킹 알고리즘 기반 SLAM 및 미로 탈출**
   - LiDAR와 IMU 데이터를 이용해 벽 정보를 추출하고, 탐색하면서 맵을 생성합니다.
2. **맵 기반 로컬리제이션 및 경로 계획**
   - 완성된 맵을 기반으로 랜덤한 위치에서 로컬리제이션을 수행하고, 주어진 목적지까지 A* 알고리즘으로 이동합니다.
3. **시뮬레이션 환경 지원**
   - Gazebo 또는 RViz 환경에서 전체 과정을 시뮬레이션 할 수 있습니다.
4. **웹캠을 통해 교통표지판 인식**
   - 사전에 학습된 CNN모델을 이용해 프레임 내 교통 표지판을 분류합니다. 최종 결과값을 이용하여 로봇의 주행속도를 조절합니다.
     
## 사용 영상

### 백트레킹 영상
[Screencast from 06-15-2025 08:20:04 PM.webm](https://github.com/user-attachments/assets/ef959bc0-720d-4071-844d-05d92196cdda)

### 로컬리제이션 영상




## 사용 방법

## 💻 환경 설정

### 1. Ubuntu 22.04 LTS 설치 (Remote PC)

Ubuntu 22.04 LTS Desktop 이미지를 아래 링크에서 다운로드하여 설치합니다.

- [Ubuntu 22.04 LTS Desktop image (64-bit)](https://releases.ubuntu.com/22.04/)
- 설치 방법: [Install Ubuntu Desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop)

---

### 2. ROS 2 Humble 설치 (Remote PC)

공식 ROS 2 문서를 참고하여 ROS 2 Humble을 설치합니다.

- [ROS 2 Humble 설치 가이드](https://docs.ros.org/en/humble/Installation.html)
- 대부분의 Linux 사용자에게는 **Debian 패키지 설치 방법**을 권장합니다.


### 1. 패키지 설치

```bash
# 워크스페이스 생성 및 빌드
mkdir -p ~/maze_ws/src
cd ~/maze_ws/src
git clone https://github.com/yu-nsojeong/robot_programing_project.git
cd ..
rosdep install -i --from-path src --rosdistro humble -y

colcon build  --packages-select find_maze_interfaces
source install/setup.bash

colcon build --symlink-install
source install/setup.bash
```
## 2. 패키지 실행

실행시 imu 패키지에서 imu 값이 셋팅되므로 터틀봇을 잘 위치시킨 후 imu 노드 및 런치를 실행시키기 바랍니다.

## 2-1시뮬레이션
###시뮬레이션 가제보 실행
```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py 
```
### 시뮬레이션 localization
```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py 
```
```bash
$  ros2 launch find_maze localization_nav_notclient.launch.py 
```
```bash
$  ros2 run find_maze kac 
```

### 시뮬레이션 back tracking
```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py 
```
```bash
$ ros2 launch find_maze right_wall_followerback.launch.py
```

##실제 환경
## 2-2. 터틀봇 브링업 (실제 로봇 활용)
터틀봇 bring up을 비롯한 로봇및 pc 초기 setting 설치방법은 아래를 참고하세요.
- 설치 방법: [Quick start guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)


### 2-2-1. SBC 원격 접속
```bash
$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
```
### 2-2-2. ROS_DOMAIN_ID 통일(ssh 접속 후) 
```bash
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_bringup robot.launch.py
```
### 2-2-3. RMW 구현 통일(내 pc)
```bash
$ export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```
### 2-3-4. ROS_DOMAIN_ID 통일(내 pc 및 ssh)
```bash
$ export ROS_DOMAIN_ID=30
```

```bash
$ ros2 topic list

#토픽 확인하여 아래와 같이 뜨면 잘 된 것입니다. (ssh 접속하지 않은 내 컴퓨터에서)
ros2 topic list 
/battery_state
/cmd_vel
/imu
...

```

### localization
```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py 
```
```bash
$  ros2 launch find_maze localization_nav_notclient.launch.py 
```
```bash
$  ros2 run find_maze kac 
```

### back tracking
```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_world_maze.launch.py 
```
```bash
$ ros2 launch find_maze right_wall_followerback.launch.py
```

### Computer vision
```bash
$  ros2 run find_maze vision
```
```bash
$ ffmpeg -f v4l2 -i /dev/video0 \
-s 1200x800 -r 15 \
-vcodec libx264 -preset ultrafast -tune zerolatency -crf 28 \
-f mpegts udp://172.100.2.21:5001  #개인 PC환경에 따라 다름
```

