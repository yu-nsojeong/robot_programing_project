# 🐢 TurtleBot3 미로 탈출 프로젝트

TurtleBot3와 ROS 2를 이용해 미로를 탐색하고, 탈출하며, 목표 지점까지 스스로 도달하는 프로젝트입니다.  
SLAM, Localization, A* 경로 계획 등을 통해 자율적으로 미로 환경을 이해하고 주행할 수 있습니다.  
시뮬레이션 환경에서도 동일한 기능을 테스트할 수 있습니다.

## 📌 주요 기능

1. **오른쪽 우선 백트래킹 알고리즘 기반 SLAM 및 미로 탈출**
   - LiDAR와 IMU 데이터를 이용해 벽 정보를 추출하고, 탐색하면서 맵을 생성합니다.
2. **맵 기반 로컬리제이션 및 경로 계획**
   - 완성된 맵을 기반으로 랜덤한 위치에서 로컬리제이션을 수행하고, 주어진 목적지까지 A* 알고리즘으로 이동합니다.
3. **시뮬레이션 환경 지원**
   - Gazebo 또는 RViz 환경에서 전체 과정을 시뮬레이션 할 수 있습니다.

## 🚀 사용 방법

### 1. 패키지 설치

```bash
# ROS 2 Humble 기반 환경 준비
sudo apt update
sudo apt install ros-humble-desktop

# 워크스페이스 생성 및 빌드
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src
git clone <이 리포지토리 주소>
cd ..
rosdep install -i --from-path src --rosdistro humble -y
colcon build --symlink-install
