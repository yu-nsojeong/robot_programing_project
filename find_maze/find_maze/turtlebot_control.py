import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from find_maze_interfaces.srv import Move  # srv/MoveCommand.srv
from std_msgs.msg import Int32
import statistics  # 중앙값 계산용


import time
import math


class MoveServiceServer(Node):
    def __init__(self):
        super().__init__("move_service_server")
        self.srv = self.create_service(Move, "move_robot", self.handle_request)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("이동 서비스 서버 시작됨!")

        self.speed_list = []
        self.speed = 0.2
        self.speed_finalized = False

        self.subscription = self.create_subscription(
            Int32, "traffic_sign_result", self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        if self.speed_finalized:
            return  # 이미 결정되었으면 더 이상 처리 안 함

        value = msg.data / 100 / 7 * 2  # 터틀봇 최대 속도 맞춰주기
        self.speed_list.append(value)
        self.speed = value

        if len(self.speed_list) == 50:

            self.speed = statistics.median(self.speed_list)
            self.speed_finalized = True
        else:
            pass

    def move_forward(self, speed, distance=0.4):

        print(f"self.speed{speed}")
        duration = distance / speed
        twist = Twist()
        twist.linear.x = speed * distance / abs(distance)
        self.pub.publish(twist)
        time.sleep(abs(duration))
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
        self.get_logger().info(f"회전각도 degree = {request.degree}")
        self.get_logger().info(f"이동거리 distance = {request.distance}")

        # back 일대는 먼저 뒤로 가야 하기 때문
        # if request.distance < 0 :
        #     self.move_forward(speed= -1 * self.speed, distance=request.distance)
        #     time.sleep(1)

        self.rotate(angular_speed=0.5, degree=request.degree)
        time.sleep(1)
        # 전진

        # if request.distance >= 0.0 :
        self.move_forward(speed=self.speed, distance=request.distance)
        time.sleep(1)

        response.success = True
        response.message = "이동 완료!"
        return response


def main():
    rclpy.init()
    node = MoveServiceServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
