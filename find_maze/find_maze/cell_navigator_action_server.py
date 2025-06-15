import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from find_maze_interfaces.action import NavigateToCell  # 사용자 정의 액션 인터페이스
from find_maze_interfaces.srv import Move

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from find_maze.cell_localizer_utils import (
    CellMapGenerator,
    AStarCellPlanner,
    extract_wall_info_from_scan_with_yaw,
)
from rclpy.executors import MultiThreadedExecutor
import math
from std_msgs.msg import Float32

import time
from ament_index_python.packages import get_package_share_directory
import os

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class CellNavigatorActionServer(Node):
    def __init__(self):
        super().__init__("cell_navigator_action_server")

        # 맵 로딩 및 A* 경로 탐색 준비

        package_share_directory = get_package_share_directory('find_maze')

        # self.declare_parameter("yaml_path", "/home/yoon/map.yaml")
        # self.declare_parameter("pgm_path", "/home/yoon/map.pgm")

        # self.yaml_path = (
        #     self.get_parameter("yaml_path").get_parameter_value().string_value
        # )
        # self.pgm_path = (
        #     self.get_parameter("pgm_path").get_parameter_value().string_value
        # )
                # 패키지 share 디렉토리 경로를 미리 얻어둡니다.
        package_share_directory = get_package_share_directory('find_maze')

        # --- 파라미터 선언 및 기본값 설정 (수정된 부분) ---
        # 기본값을 패키지 내 상대 경로로 설정합니다.
        default_yaml_path = os.path.join(package_share_directory, 'map', 'map.yaml')
        default_pgm_path = os.path.join(package_share_directory, 'map', 'map.pgm')

        self.declare_parameter("yaml_path", default_yaml_path)
        self.declare_parameter("pgm_path", default_pgm_path)

        # 파라미터 값 가져오기
        self.yaml_path = self.get_parameter("yaml_path").get_parameter_value().string_value
        self.pgm_path = self.get_parameter("pgm_path").get_parameter_value().string_value

        #self.get_logger().info(f"로딩할 YAML 맵 경로: {self.yaml_path}")
        #self.get_logger().info(f"로딩할 PGM 맵 경로: {self.pgm_path}")

        # 이제 self.yaml_path와 self.pgm_path를 사용하여 맵 파일을 로드하면 됩니다.
        try:
            # 예시: YAML 파일을 로드하는 코드 (PyYAML 라이브러리가 필요할 수 있습니다)
            # import yaml
            # with open(self.yaml_path, 'r') as file:
            #     map_data = yaml.safe_load(file)
            # self.get_logger().info(f"맵 데이터 로드 완료 (YAML): {map_data}")

            # 예시: PGM 파일을 로드하는 코드 (필요하다면)
            # with open(self.pgm_path, 'rb') as file: # 이진 모드로 열기
            #     pgm_data = file.read()
            # self.get_logger().info(f"맵 데이터 로드 완료 (PGM): {len(pgm_data)} bytes")

            pass # 실제 맵 로드 로직이 이곳에 위치

        except Exception as e:
            self.get_logger().error(f"맵 파일 로드 중 오류 발생: {e}")

        # self.map_gen = CellMapGenerator('/home/yoon/map.yaml', '/home/yoon/map.pgm')
        self.map_gen = CellMapGenerator(self.yaml_path, self.pgm_path)
        self.cell_map = self.map_gen.get_cell_map()
        self.planner = AStarCellPlanner(self.cell_map)

        # 현재 위치 후보군 (초기에는 전체 셀)
        self.estimated_possibilities = list(self.cell_map.keys())

        # 로봇 이동을 위한 cmd_vel 퍼블리셔
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)

        lidar_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # 신뢰성 정책: BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,         # 기록 정책: 가장 최근 메시지 유지
            depth=1,                                 # 큐 깊이: 1 (가장 최신 스캔만 중요)
            durability=DurabilityPolicy.VOLATILE     # 지속성 정책: 휘발성 (과거 메시지 전달 안 함)
        )
        # 라이다 센서 구독 (장애물/벽 감지)
        self.scan_sub = self.create_subscription(

            LaserScan, "scan", self.lidar_callback,lidar_qos_profile
        )

        # 액션 서버 생성: NavigateToCell 액션을 처리함
        self._action_server = ActionServer(
            self, NavigateToCell, "navigate_to_cell", self.execute_callback
        )
        # '/imu_yaw_deg' 토픽 구독
        self.subscription = self.create_subscription(
            Float32, "/imu_yaw_deg", self.imuCallback, 10
        )
        self.dimu = 0
        self.busy = False

        # 서비스 클라이언트 설정
        self.cli = self.create_client(Move, "move_robot")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("🛠 서비스 기다리는 중...")

        # 주기적으로 위치 추정 및 경로 탐색 시도 (0.3초마다)
        #print("set timer")
        self.timer = self.create_timer(0.3, self.process_estimation)
        self.last_scan_msg = None  # 최신 라이다 메시지 저장용

        # 내부 상태 변수 초기화
        self.active_goal = None  # 현재 처리 중인 goal 핸들
        self.feedback_cb = None  # 피드백 콜백 (현재는 사용 안함)
        self.path = []  # A* 경로
        self.current_index = 0  # 현재 경로에서의 인덱스 위치

        self.last_wall_info = None  # 이전 라이다 기반 벽 정보
        self.last_move_direction = None  # 마지막으로 이동한 방향 ('up', 'down', ...)
        self.go_degree = None
        self.isWhile = True
        self.feedback_msg = None
        self.goal_handle = None

        self.grid_size_max = 5

    def imuCallback(self, msg):
        #self.get_logger().info(f"📥 받은 Yaw(deg): {msg.data:.2f} 도")
        self.dimu = msg.data

    def process_estimation(self):
        #print("process ectimation")

        #!아직 목표가 안 왔다면 실행 x
        if self.active_goal is None or self.last_scan_msg is None:
            return

        if self.busy or not self.isWhile:
            return

        #print("process ectimation2")

        #!라이다에서 벽 정보 추출
        wall_info = extract_wall_info_from_scan_with_yaw(self.last_scan_msg,self.dimu)
        #print("WALL _ INFO :", wall_info)

        #!예측 지점 후보군 생성
        self.filter_estimated_cells(wall_info)

        # #!과거 방향 고려해 후보군 더 줄이기
        # if self.last_wall_info and self.last_move_direction:
        #     self.filter_by_past_info(self.last_wall_info, self.last_move_direction)




        #!후보 셀이 하나로 확정되었고 아직 경로 없음 => 경로 생성
        if len(self.estimated_possibilities) == 1 and not self.path:
            start = self.estimated_possibilities[0]
            goal = (
                self.active_goal.request.target_row,
                self.active_goal.request.target_col,
            )
            self.path = self.planner.plan(start, goal)
            self.current_index = 0

        #!이제 현재 좌표에 따라 어디로 가야 될 지 정하기
        if self.path:

            #!도착!
            if self.current_index + 1 > len(self.path) :
                #print("도도착착")
                self.isWhile = False
                return
            current_cell = self.path[self.current_index]

            if current_cell[0] == self.goal_handle.request.target_row and current_cell[1] == self.goal_handle.request.target_col :
                #print("도도착착착")
                self.isWhile = False
                return


            next_cell = self.path[self.current_index + 1]

            dr = next_cell[0] - current_cell[0]
            dc = next_cell[1] - current_cell[1]

            # 오른쪽 yaw 0 기준 계산
            if dr == -1 and dc == 0:
                self.go_degree = 90  # up
            elif dr == 1 and dc == 0:
                self.go_degree = -90  # down
            elif dr == 0 and dc == -1:
                self.go_degree = 180  # left
            elif dr == 0 and dc == 1:
                self.go_degree = 0  # right
            else:
                pass

            # 현재 각도를 기준으로 실제로 움직여야 하는 각도 계산
            self.go_degree = self.go_degree - self.dimu
            if self.go_degree > 270:
                self.go_degree - 360
            elif self.go_degree < -270:
                self.go_degree + 360

            self.current_index += 1

            #피드백 메시지 보내기
            #print(f"{self.feedback_msg.current_row}  {self.feedback_msg.current_col}")
            self.feedback_msg.current_row = current_cell[0]
            self.feedback_msg.current_col = current_cell[1]
            self.goal_handle.publish_feedback(self.feedback_msg)
        #     self.feedback_msg.current_col = i
        #     self.goal_handle.publish_feedback(self.feedback_msg)
        #     self.get_logger().info(f"📤 피드백: {i}")

        #!후보 셀이 하나가 아니라면 갈 수 있는 방향 중 goal에 가까운 방향으로 가도록 하기
        elif len(self.estimated_possibilities) > 1 and not self.path:
            # 후보가 많으면 휴리스틱 이동
            goal = (
                self.active_goal.request.target_row,
                self.active_goal.request.target_col,
            )
            self.last_move_direction = self.choose_best_direction(goal)
            self.go_degree = self.choose_yaw_direction(self.last_move_direction,self.dimu)
            self.update_estimated_possibilities_after_move(self.last_move_direction)
            # 3. 마지막 벽 정보 저장
            self.last_wall_info = wall_info




        self.send_move_command(self.go_degree,0.4)

    def send_move_command(self, degree, distance):
        self.busy = True

        if hasattr(self, "reset_timer"):
            self.reset_timer.cancel()
            del self.reset_timer

        req = Move.Request()
        req.degree = float(degree)
        req.distance = float(distance)

        self.get_logger().info(f"📤 요청: degree={degree}, distance={distance}")

        future = self.cli.call_async(req)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        """
        서비스 응답 처리 - 성공 여부 출력, 일정 시간 후 busy 해제 예약
        """
        try:
            result = future.result()
            self.get_logger().info(
                f"✅ 서비스 응답: success={result.success}, message={result.message}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 호출 실패: {e}")
        finally:
            # 이전 타이머가 있다면 취소하고 제거
            if hasattr(self, "reset_timer"):
                self.reset_timer.cancel()
                del self.reset_timer

            self.get_logger().info("⏳ 2초 후 busy 해제 타이머 설정")
            self.reset_timer = self.create_timer(2.0, self.reset_busy_once)

    def reset_busy_once(self):
        """
        busy 상태 해제 콜백
        """
        if self.busy:

            self.get_logger().info("🟢 다음 명령 대기 가능")
            self.busy = False


    def choose_yaw_direction(self, direction, dyaw):
        direction_map = {
            "up": 90.0,
            "down": -90.0,
            "left": 180.0,
            "right": 0.0,
        }
        rotate_degree = 0
        rotate_degree = direction_map[direction] - dyaw
        if rotate_degree > 270 :
            rotate_degree += -360
        elif rotate_degree < -270:
            rotate_degree += 360


        return rotate_degree







    def choose_best_direction(self, goal):
        """
        후보가 여러 개인 경우, goal에 가까워지는 방향으로 이동.
        이동한 방향은 self.last_move_direction에 기록.
        """
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        best_direction = None
        min_distance = float("inf")

        for cell in self.estimated_possibilities:
            for dir, (dr, dc) in direction_map.items():
                if not self.cell_map[cell][dir]:
                    continue  # 해당 방향으로 못 가면 스킵

                next_cell = (cell[0] + dr, cell[1] + dc)
                if next_cell not in self.cell_map:
                    continue

                # 맨해튼 거리 기준으로 goal과 가장 가까워지는 방향 선택
                dist = abs(next_cell[0] - goal[0]) + abs(next_cell[1] - goal[1])
                if dist < min_distance:
                    min_distance = dist
                    best_direction = dir

        if best_direction:
            # self.last_move_direction = best_direction
            self.get_logger().info(
                f"↪️ 후보가 여러 개일 때, {best_direction} 방향으로 이동"
            )
            return best_direction

    def lidar_callback(self, msg):
        # goal 없으면 무시
        # if self.active_goal is None:
        #     return

        #print("lidar")

        if not self.feedback_msg :
            return
        #print("aslkdfjklasjdfklasj")
        self.last_scan_msg = msg

        # for i in range(1, 101):
        #     # 피드백 채워서 전송
        #     self.feedback_msg.current_row = i
        #     self.feedback_msg.current_col = i
        #     self.goal_handle.publish_feedback(self.feedback_msg)
        #     self.get_logger().info(f"📤 피드백: {i}")
        #     time.sleep(0.05)  # 50ms 간격

        #self.isWhile = False

    def filter_estimated_cells(self, wall_info):
        """
        현재 라이다로 얻은 벽 정보(wall_info)를 바탕으로
        self.estimated_possibilities 를 업데이트한다.
        """

        self.get_logger().info(f"DEBUG: 현재 self.cell_map 내용: {self.cell_map}")
        self.get_logger().info(f"DEBUG: 비교 대상 wall_info: {wall_info}")
        self.estimated_possibilities = [
            cell
            for cell in self.estimated_possibilities
            if self.cell_map[cell] == wall_info
        ]
        self.get_logger().info(f"🔍 후보 셀 수: {len(self.estimated_possibilities)}")

    def update_estimated_possibilities_after_move(self, direction):
        """
        self.estimated_possibilities의 모든 셀에 (delta_row, delta_col)을 더하고,
        유효 범위 (0-4)를 벗어나는 셀은 제외하여 리스트를 업데이트(덮어쓰기)합니다.

        Args:
            delta_row (int): 각 셀의 행(row)에 더할 값.
            delta_col (int): 각 셀의 열(col)에 더할 값.
        """
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        delta_row, delta_col = direction_map[direction]

        self.get_logger().info(f"이동 전 예상 위치 후보: {self.estimated_possibilities}")

        # 리스트 컴프리헨션을 사용하여 새로운 셀 목록을 생성하고,
        # 이를 self.estimated_possibilities에 직접 할당(덮어쓰기)합니다.
        self.estimated_possibilities = [
            (cell[0] + delta_row, cell[1] + delta_col) # 새 (row, col) 튜플 생성
            for cell in self.estimated_possibilities # 현재 리스트의 각 셀에 대해
            if (0 <= (cell[0] + delta_row) < self.grid_size_max and # 새 행이 유효 범위 내에 있고
                0 <= (cell[1] + delta_col) < self.grid_size_max)    # 새 열이 유효 범위 내에 있을 때만 포함
        ]

        self.get_logger().info(f"이동 후 업데이트된 예상 위치 후보: {self.estimated_possibilities}")
        self.get_logger().info(f"업데이트된 후보 셀 수: {len(self.estimated_possibilities)}")

    def filter_by_past_info(self, past_wall_info, direction):
        """
        이전 라이다 정보와 이동 방향을 기반으로 현재 후보 셀을 더 정밀하게 걸러낸다.
        """
        reverse_dir = {"up": "down", "down": "up", "left": "right", "right": "left"}
        rev_dir = reverse_dir[direction]

        filtered = []
        for cell in self.estimated_possibilities:
            # 해당 셀의 rev_dir 방향에 벽이 뚫려 있어야 하고 <- 당연히 뚤려 있겠지 다 그쪽으로 갈 수 있는 애들만 추린건데
            if not self.cell_map[cell][rev_dir]:
                continue

            # 해당 셀에서 rev_dir 쪽으로 한 칸 이동한 과거 셀의 wall_info가 일치해야 함
            delta_map = {
                "up": (1, 0),
                "down": (-1, 0),
                "left": (0, 1),
                "right": (0, -1),
            }
            dr, dc = delta_map[direction]
            past_cell = (cell[0] + dr, cell[1] + dc)

            if (
                past_cell in self.cell_map
                and self.cell_map[past_cell] == past_wall_info
            ):
                filtered.append(cell)

        if filtered:
            self.estimated_possibilities = filtered
            self.get_logger().info(
                f"🧠 과거 정보로 후보 셀 줄임 → {len(filtered)}개 {filtered}"
            )

    def execute_callback(self, goal_handle):
        # 목표 좌표 로그 출력
        self.get_logger().info(
            f"📍 목표 셀: ({goal_handle.request.target_row}, {goal_handle.request.target_col})"
        )


        self.active_goal = goal_handle

        # self.active_goal.request.target_row = goal_handle.request.target_row
        # self.active_goal.request.target_col = goal_handle.request.target_col



        self.goal_handle = goal_handle
        self.feedback_msg = NavigateToCell.Feedback()

        #도착 할 때 까지 블로킹
        while self.isWhile :
            pass


        # 결과 전송
        result = NavigateToCell.Result()
        result.success = True
        result.message = "100까지 카운트 완료!"
        goal_handle.succeed()
        return result

    def stop_robot(self):
        # 로봇 정지 명령 전송
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = CellNavigatorActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()
