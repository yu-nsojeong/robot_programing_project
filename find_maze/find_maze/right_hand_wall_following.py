import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from transforms3d.euler import quat2euler
from find_maze_interfaces.srv import Move  # 커스텀 서비스
from rclpy.qos import qos_profile_sensor_data


class GlobalLidarController(Node):
    def __init__(self):
        super().__init__('lidar_service_controller')

        # 내부 상태 초기화
        self.busy = False  # 현재 서비스 요청 중인지 여부
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.front_mean = 0.0  # 라이다로 측정한 전방 평균 거리
        self.right_mean = 0.0  # 라이다로 측정한 오른쪽 평균 거리

        self.FORWARD = False

        # 센서 데이터 구독
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)#SensorDataQoS()는 RELIABILITY=BestEffort, HISTORY=KeepLast로 맞춰져 있어 LiDAR와 잘 붙음.
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

        # 서비스 클라이언트 설정
        self.cli = self.create_client(Move, 'move_robot')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('🛠 서비스 기다리는 중...')

        # 주기적으로 판단하는 타이머 (0.5초 간격)
        self.create_timer(0.5, self.decision_loop)

    def scan_callback(self, msg):
        """
        라이다 콜백 - 센서 데이터를 받아 front_mean, right_mean만 업데이트함
        판단과 명령은 decision_loop에서 수행
        """
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        front_sum = 0
        front_size = 0
        right_sum = 0
        right_size = 0

        for x, y, r in zip(x_local, y_local, ranges):
            if x > 0 and abs(y) < abs(x):  # 전방
                front_sum += x
                front_size += 1
            elif y < 0 and abs(x) < abs(y):  # 오른쪽
                right_sum += y
                right_size += 1

        # 평균 거리 계산
        self.front_mean = abs(front_sum / front_size) if front_size > 0 else 0.0
        self.right_mean = abs(right_sum / right_size) if right_size > 0 else 0.0

    def odom_callback(self, msg):
        """
        오도메트리 콜백 - 로봇 위치 및 방향 업데이트
        """
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        quat = [ori.w, ori.x, ori.y, ori.z]
        _, _, yaw = quat2euler(quat)
        self.robot_x = pos.x
        self.robot_y = pos.y
        self.robot_yaw = yaw




    def decision_loop(self):
        if self.busy:
            return

        self.get_logger().info(f"📌 판단: front={self.front_mean:.2f}, right={self.right_mean:.2f}")

        # if self.FORWARD:
        #     self.get_logger().info("⬆️ 오른쪽으로 전진 요청")
        #     self.FORWARD = False
        #     self.send_move_command(degree=0.0, distance=0.4)
        if self.right_mean > 0.28:
            self.get_logger().info("➡️ 오른쪽 회전 요청")
            self.send_move_command(degree=-90.0, distance=0.4)
            self.FORWARD = True
        elif self.front_mean > 0.28:
            self.get_logger().info("⬆️ 전진 요청")
            self.send_move_command(degree=0.0, distance=0.4)
        else:
            self.get_logger().info("⬅️ 왼쪽 회전 요청")
            self.send_move_command(degree=90.0, distance=0.0)

    def send_move_command(self, degree, distance):
        self.busy = True

        if hasattr(self, 'reset_timer'):
            self.reset_timer.cancel()
            del self.reset_timer

        req = Move.Request()
        req.degree = float(degree)
        req.distance = float(distance)

        self.get_logger().info(f"📤 요청: degree={degree}, distance={distance}")

        future = self.cli.call_async(req)
        future.add_done_callback(self.handle_response)

    # def decision_loop(self):
    #     """
    #     판단 루프 - 일정 주기로 front_mean, right_mean을 기반으로 명령 내림
    #     busy 상태면 무시
    #     """
    #     if self.busy:
    #         return

    #     self.get_logger().info(f"📌 판단: front={self.front_mean:.2f}, right={self.right_mean:.2f}")

    #     # 거리 판단에 따라 명령 선택
    #     if self.FORWARD :
    #         self.get_logger().info("⬆️  오른쪽 전진 요청")
    #         self.FORWARD = False
    #         self.send_direction_request("FORWARD")
    #     elif self.right_mean > 0.28:
    #         self.get_logger().info("➡️⬆️  오른쪽 회전 ")
    #         self.send_direction_request("RIGHT")
    #         self.FORWARD = True
    #     elif self.front_mean > 0.28:
    #         self.get_logger().info("⬆️  전진 요청")
    #         self.send_direction_request("FORWARD")
    #     else:
    #         self.get_logger().info("⬅️  왼쪽 회전 요청")
    #         self.send_direction_request("LEFT")

    # def send_direction_request(self, direction):
    #     """
    #     서비스 요청을 보내고 busy 상태로 변경
    #     """
    #     self.busy = True

    #         # 이전 타이머가 있으면 취소
    #     if hasattr(self, 'reset_timer'):
    #         self.reset_timer.cancel()
    #         del self.reset_timer

    #     req = Move.Request()
    #     req.direction = direction
    #     print(req.direction)
    #     print(f"[디버그] direction 값: {direction} / 타입: {type(direction)}")

    #     future = self.cli.call_async(req)
    #     future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        """
        서비스 응답 처리 - 성공 여부 출력, 일정 시간 후 busy 해제 예약
        """
        try:
            result = future.result()
            self.get_logger().info(f"✅ 서비스 응답: success={result.success}, message={result.message}")
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 호출 실패: {e}")
        finally:
            # 이전 타이머가 있다면 취소하고 제거
            if hasattr(self, 'reset_timer'):
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

def main():
    rclpy.init()
    node = GlobalLidarController()
    try:
        rclpy.spin(node)  # 노드 실행
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
