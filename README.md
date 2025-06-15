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
colcon build --symlink-install
```

### 2.. 터틀봇 브링업 실행
터틀봇 bring up을 비롯한 로봇및 pc 초기 setting 설치방법은 아래를 참고하세요.
- 설치 방법: [Quick start guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
