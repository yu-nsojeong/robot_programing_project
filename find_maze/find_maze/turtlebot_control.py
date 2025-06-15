import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from find_maze_interfaces.srv import Move  # srv/MoveCommand.srv

import time
import math

class MoveServiceServer(Node):
    def __init__(self):
        super().__init__('move_service_server')
        self.srv = self.create_service(Move, 'move_robot', self.handle_request)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('🚀 이동 서비스 서버 시작됨!')

    def move_forward(self, speed=0.1, distance=0.4):
        duration = distance / speed
        twist = Twist()
        twist.linear.x = speed
        self.pub.publish(twist)
        time.sleep(duration)
        twist.linear.x = 0.0
        self.pub.publish(twist)

    def rotate(self, angular_speed=0.5, degree=90):
        # degree → radian 변환
        radians = math.radians(degree)
        duration = abs(radians) / angular_speed

        twist = Twist()
        twist.angular.z = angular_speed if degree > 0 else -angular_speed
        self.pub.publish(twist)
        time.sleep(duration)
        twist.angular.z = 0.0
        self.pub.publish(twist)



    def handle_request(self, request, response):
        self.get_logger().info(f"📥 회전각도 degree = {request.degree}")
        self.get_logger().info(f"📥 이동거리 distance = {request.distance}")

        # 회전 먼저
        self.rotate(angular_speed=0.5, degree=request.degree)
        time.sleep(1)
        # 전진
        self.move_forward(speed=0.1, distance=request.distance)
        time.sleep(1)

        response.success = True
        response.message = '이동 완료!'
        return response

    # def handle_request(self, request, response):
    #     self.get_logger().info(f"📥 request.direction = '{request.direction}'")
    #     cmd = request.direction.lower()
    #     if cmd == 'forward':
    #         self.get_logger().info('⬆️ 전진 요청 받음')
    #         self.move_forward()
    #         response.success = True
    #         response.message = 'Moved forward'
    #     elif cmd == 'backward':
    #         self.get_logger().info('⬇️ 후진 요청 받음')
    #         self.rotate(angular_speed=0.5, duration=3.14 / 0.5)
    #         self.move_forward()
    #         response.success = True
    #         response.message = 'Moved backward (after 180 turn)'
    #     elif cmd == 'left':
    #         self.get_logger().info('⬅️ 왼쪽 회전 요청 받음')
    #         self.rotate(angular_speed=0.5)
    #         response.success = True
    #         response.message = 'Turned left'
    #     elif cmd == 'right':
    #         self.get_logger().info('➡️ 오른쪽 회전 요청 받음')
    #         self.rotate(angular_speed=-0.5)
    #         response.success = True
    #         response.message = 'Turned right'
    #     else:
    #         response.success = False
    #         response.message = f'알 수 없는 명령: {cmd}'
    #     self.get_logger().info(f"📨 응답 완료: success={response.success}, message='{response.message}'")

    #     time.sleep(1)
    #     return response

def main():
    rclpy.init()
    node = MoveServiceServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
