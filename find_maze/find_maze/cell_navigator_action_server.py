import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from find_maze_interfaces.action import NavigateToCell
from find_maze_interfaces.srv import Move

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from find_maze.cell_localizer_utils import (
    CellMapGenerator,
    AStarCellPlanner,
    extract_wall_info_from_scan_with_yaw,
)
from rclpy.executors import MultiThreadedExecutor
import os
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory


class CellNavigatorActionServer(Node):
    def __init__(self):
        super().__init__("cell_navigator_action_server")

        package_share_directory = get_package_share_directory("find_maze")
        default_yaml_path = os.path.join(package_share_directory, "map", "map.yaml")
        default_pgm_path = os.path.join(package_share_directory, "map", "map.pgm")

        self.declare_parameter("yaml_path", default_yaml_path)
        self.declare_parameter("pgm_path", default_pgm_path)

        self.yaml_path = (
            self.get_parameter("yaml_path").get_parameter_value().string_value
        )
        self.pgm_path = (
            self.get_parameter("pgm_path").get_parameter_value().string_value
        )

        try:
            pass  # 실제 맵 로드 로직 위치
        except Exception as e:
            self.get_logger().error(f"맵 파일 로드 중 오류 발생: {e}")

        self.map_gen = CellMapGenerator(self.yaml_path, self.pgm_path)
        self.cell_map = self.map_gen.get_cell_map()
        self.planner = AStarCellPlanner(self.cell_map)

        self.estimated_possibilities = list(self.cell_map.keys())
        # self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)

        lidar_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.lidar_callback, lidar_qos_profile
        )

        self._action_server = ActionServer(
            self,
            NavigateToCell,
            "navigate_to_cell",
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
        )

        self.subscription = self.create_subscription(
            Float32, "/imu_yaw_deg", self.imuCallback, 10
        )

        self.dimu = 0
        self.busy = False
        self.cli = self.create_client(Move, "move_robot")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("🛠 서비스 기다리는 중...")

        self.timer = self.create_timer(0.3, self.process_estimation)
        self.last_scan_msg = None

        self.active_goal = None
        self.feedback_cb = None
        self.path = []
        self.current_index = 0
        self.last_wall_info = None
        self.last_move_direction = None
        self.go_degree = None
        self.isWhile = True
        self.feedback_msg = None
        self.goal_handle = None

        self.grid_size_max = 5

    def goal_callback(self, goal_request):
        row, col = goal_request.target_row, goal_request.target_col
        if not (0 <= row < self.grid_size_max and 0 <= col < self.grid_size_max):
            self.get_logger().warn(f"❌ 목표 REJECT: ({row}, {col})")
            return GoalResponse.REJECT
        if (row, col) not in self.cell_map:
            self.get_logger().warn(f"❌ 셀맵에 없음: ({row}, {col})")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("🔄 목표 취소 요청 수락")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        row, col = goal_handle.request.target_row, goal_handle.request.target_col
        self.get_logger().info(f"📍 목표 셀: ({row}, {col})")

        # self.active_goal = goal_handle
        # self.goal_handle = goal_handle
        # self.feedback_msg = NavigateToCell.Feedback()

        # 🔧 상태 초기화
        self.active_goal = goal_handle
        self.goal_handle = goal_handle
        self.feedback_msg = NavigateToCell.Feedback()
        self.path = []
        self.current_index = 0
        self.isWhile = True
        self.last_wall_info = None
        self.last_move_direction = None
        self.go_degree = None
        self.estimated_possibilities = list(self.cell_map.keys())

        while self.isWhile:
            pass

        result = NavigateToCell.Result()
        result.success = True
        result.message = "목표지점 도착!"
        goal_handle.succeed()
        self.isWhile = True
        return result

    def imuCallback(self, msg):
        self.dimu = msg.data

    def lidar_callback(self, msg):
        if not self.feedback_msg:
            return
        self.last_scan_msg = msg

    def process_estimation(self):
        if self.active_goal is None or self.last_scan_msg is None:
            return
        if self.busy or not self.isWhile:
            return

        wall_info = extract_wall_info_from_scan_with_yaw(self.last_scan_msg, self.dimu)
        self.filter_estimated_cells(wall_info)

        if len(self.estimated_possibilities) == 1 and not self.path:
            start = self.estimated_possibilities[0]
            goal = (
                self.active_goal.request.target_row,
                self.active_goal.request.target_col,
            )
            self.path = self.planner.plan(start, goal)
            self.current_index = 0

        if self.path:
            if self.current_index + 1 > len(self.path):
                self.isWhile = False
                return

            current_cell = self.path[self.current_index]

            if current_cell == (
                self.goal_handle.request.target_row,
                self.goal_handle.request.target_col,
            ):
                self.isWhile = False
                return

            next_cell = self.path[self.current_index + 1]
            dr = next_cell[0] - current_cell[0]
            dc = next_cell[1] - current_cell[1]

            if dr == -1 and dc == 0:
                self.go_degree = 90
            elif dr == 1 and dc == 0:
                self.go_degree = -90
            elif dr == 0 and dc == -1:
                self.go_degree = 180
            elif dr == 0 and dc == 1:
                self.go_degree = 0

            self.go_degree = self.go_degree - self.dimu
            if self.go_degree > 270:
                self.go_degree -= 360
            elif self.go_degree < -270:
                self.go_degree += 360

            self.current_index += 1
            self.feedback_msg.current_row = current_cell[0]
            self.feedback_msg.current_col = current_cell[1]
            self.goal_handle.publish_feedback(self.feedback_msg)

        elif len(self.estimated_possibilities) > 1 and not self.path:
            print("333")
            goal = (
                self.active_goal.request.target_row,
                self.active_goal.request.target_col,
            )
            self.last_move_direction = self.choose_best_direction(goal)
            self.go_degree = self.choose_yaw_direction(
                self.last_move_direction, self.dimu
            )
            print(f"self.last_move_direction{self.last_move_direction}")
            print(f"godigree{self.go_degree}")
            self.update_estimated_possibilities_after_move(self.last_move_direction)
            self.last_wall_info = wall_info

        self.send_move_command(self.go_degree, 0.4)

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
        try:
            result = future.result()
            self.get_logger().info(
                f"✅ 서비스 응답: success={result.success}, message={result.message}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 호출 실패: {e}")
        finally:
            if hasattr(self, "reset_timer"):
                self.reset_timer.cancel()
                del self.reset_timer

            self.reset_timer = self.create_timer(2.0, self.reset_busy_once)

    def reset_busy_once(self):
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
        rotate_degree = direction_map[direction] - dyaw
        if rotate_degree > 270:
            rotate_degree -= 360
        elif rotate_degree < -270:
            rotate_degree += 360
        return rotate_degree

    def choose_best_direction(self, goal):
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
                    continue

                next_cell = (cell[0] + dr, cell[1] + dc)
                if next_cell not in self.cell_map:
                    continue

                dist = abs(next_cell[0] - goal[0]) + abs(next_cell[1] - goal[1])
                if dist < min_distance:
                    min_distance = dist
                    best_direction = dir

        if best_direction:
            self.get_logger().info(
                f"↪️ 후보가 여러 개일 때, {best_direction} 방향으로 이동"
            )
            return best_direction

    def update_estimated_possibilities_after_move(self, direction):
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        delta_row, delta_col = direction_map[direction]

        self.estimated_possibilities = [
            (cell[0] + delta_row, cell[1] + delta_col)
            for cell in self.estimated_possibilities
            if 0 <= (cell[0] + delta_row) < self.grid_size_max
            and 0 <= (cell[1] + delta_col) < self.grid_size_max
        ]

    def filter_estimated_cells(self, wall_info):
        self.estimated_possibilities = [
            cell
            for cell in self.estimated_possibilities
            if self.cell_map[cell] == wall_info
        ]


def main(args=None):
    rclpy.init(args=args)
    node = CellNavigatorActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()
