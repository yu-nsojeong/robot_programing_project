#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32
from find_maze_interfaces.srv import Move
from find_maze.cell_localizer_utils import extract_wall_info_from_scan_with_yaw
import math

DIRECTIONS = ['up', 'right', 'down', 'left']
DIR_TO_VEC = {
    'up': (-1, 0),
    'right': (0, 1),
    'down': (1, 0),
    'left': (0, -1)
}
DIR_TO_REVERSE = {
    'up': 'down',
    'right': 'left',
    'down': 'up',
    'left': 'right'
}

class MazeSolverNode(Node):
    def __init__(self):
        super().__init__('maze_solver')

        self.cli = self.create_client(Move, "move_robot")
        self.create_subscription(Float32, "/imu_yaw_deg", self.imu_callback, 10)
        self.create_subscription(LaserScan, "scan", self.lidar_callback, self._make_qos())

        self.imu_yaw_deg = 0.0
        self.lidar_data = None
        self.busy = False

        self.x, self.y = 0, 0
        self.curr_direction = 'right'

        self.cell_map = {}
        self.visited = set()
        self.path = []

        self.get_logger().info('>>>Maze solver ready<<<')
        self.create_timer(1.0, self.try_explore)

        self.branch_stack = []

    def _make_qos(self):
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

    def try_explore(self):
        if not self.busy and self.lidar_data:
            self.explore()

    def compute_rotation_angle(self, from_dir, to_dir):
        from_idx = DIRECTIONS.index(from_dir)
        to_idx = DIRECTIONS.index(to_dir)
        return ((to_idx - from_idx) % 4) * 90

    def explore(self):

        if self.is_goal():
          self.get_logger().info("🏁 미로 탈출 완료. 프로그램 종료")
          return


        if self.busy or self.lidar_data is None:
            return

        #! 현재 좌표 정하기
        curr_pos = (self.x, self.y)
                # 벽 정보가 없으면 추출
        if curr_pos not in self.cell_map:
            self.cell_map[curr_pos] = extract_wall_info_from_scan_with_yaw(self.lidar_data, self.imu_yaw_deg)
            self.get_logger().info(f"{curr_pos} 벽 정보: {self.cell_map[curr_pos]}")


        # 처음 도착한 경우 방문 기록
        if curr_pos not in self.visited:
            self.visited.add(curr_pos)

        #!갈 수 있는 방향들 중, 아직 안 가본 곳 찾기 (오른손 순서: 우→하→좌→상)
        unvisited_dirs = []
        for dir_name in ['right', 'down', 'left', 'up']:
            if self.cell_map[curr_pos].get(dir_name):  # 벽이 없고
                dx, dy = DIR_TO_VEC[dir_name]
                nx, ny = self.x + dx, self.y + dy
                if (nx, ny) not in self.visited:
                    #self.get_logger().info(f"{curr_pos} 간적 없는곳 정보: {nx}, {ny}")
                    unvisited_dirs.append((dir_name, nx, ny))


        if not unvisited_dirs:
            # 갈 수 있는 곳이 없으면 백트래킹
            self.get_logger().info(f"더 갈 곳 없음. 현재 위치: {curr_pos}")
            if self.branch_stack:
                stack_dir = self.branch_stack.pop()

                reversed_direction = DIR_TO_REVERSE[stack_dir]
                dx, dy = DIR_TO_VEC[reversed_direction]
                self.x ,self.y  = self.x + dx,  self.y + dy
                self.send_move_command(self.choose_yaw_direction(stack_dir,self.imu_yaw_deg),-0.4)
                print(self.branch_stack)

            else:
                self.get_logger().info("🏁 미로 탐색 완료. 프로그램 종료")
                rclpy.shutdown()
            return

        if len(unvisited_dirs) > 1:
            print(f"갈 수 있는 곳 개수 : {len(unvisited_dirs)}")
            #!갈수 있는 방향 중 하나 지나가야지. 그러면서 마킹하고 분기점 굳이 적어줄 필요 없을듯?

        go_direction, self.x ,self.y  = unvisited_dirs[0]
        self.branch_stack.append(go_direction)
        self.send_move_command(self.choose_yaw_direction(go_direction,self.imu_yaw_deg),0.4)
        print(self.branch_stack)



    def is_goal(self):
        ranges = self.lidar_data.ranges
        sumall = sum(ranges)



        total = len(ranges)
        inf_count = sum(1 for r in ranges if math.isinf(r))
        print(f"sum   {inf_count >= total / 2}")

        if inf_count >= total / 2:
            return True

        return False

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

    def imu_callback(self, msg):
        self.imu_yaw_deg = msg.data

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def send_move_command(self, degree, distance):
        if self.busy:
            return

        self.busy = True

        if hasattr(self, 'reset_timer'):
            self.reset_timer.cancel()
            del self.reset_timer

        req = Move.Request()
        req.degree = float(degree)
        req.distance = float(distance)

        self.get_logger().info(f"이동 요청: degree={degree}, distance={distance}")
        future = self.cli.call_async(req)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"응답: success={result.success}, message={result.message}")
        except Exception as e:
            self.get_logger().error(f"서비스 실패: {e}")
        finally:

            if hasattr(self, 'reset_timer'):
                self.reset_timer.cancel()
                del self.reset_timer

            self.get_logger().info("2초 후 busy 해제 타이머 설정")
            self.reset_timer = self.create_timer(2.0, self.reset_busy_once)

    def reset_busy_once(self):
        self.busy = False
        #self.get_logger().info("다음 명령 가능")

def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
