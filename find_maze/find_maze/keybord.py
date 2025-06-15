import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, tty, termios
import time

class TurtleBotShortMove(Node):
    def __init__(self):
        super().__init__('turtlebot_short_move')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("🕹 방향키 조작: ↑ 30cm / ↓ 180도 후 30cm / ← 90도 회전 / → 90도 회전")

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch1 = sys.stdin.read(1)
            if ch1 == '\x1b':  # ESC
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    if ch3 == 'A': return 'UP'
                    elif ch3 == 'B': return 'DOWN'
                    elif ch3 == 'C': return 'RIGHT'
                    elif ch3 == 'D': return 'LEFT'
            return ch1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def move_forward(self, speed=0.1, duration=4.0):
        twist = Twist()
        twist.linear.x = speed
        self.pub.publish(twist)
        time.sleep(duration)
        twist.linear.x = 0.0
        self.pub.publish(twist)

    def rotate(self, angular_speed=0.5, duration=3.14 / 2 / 0.5):
        twist = Twist()
        twist.angular.z = angular_speed
        self.pub.publish(twist)
        time.sleep(duration)
        twist.angular.z = 0.0
        self.pub.publish(twist)

    def run(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'UP':
                self.get_logger().info("⬆️  전진 30cm")
                self.move_forward()
            elif key == 'DOWN':
                self.get_logger().info("⬇️  180도 회전 후 전진 30cm")
                self.rotate(angular_speed=0.5, duration=3.14 / 0.5)
                self.move_forward()
            elif key == 'LEFT':
                self.get_logger().info("⬅️  왼쪽 90도 회전")
                self.rotate(angular_speed=0.5, duration=3.14 / 2 / 0.5)
            elif key == 'RIGHT':
                self.get_logger().info("➡️  오른쪽 90도 회전")
                self.rotate(angular_speed=-0.5, duration=3.14 / 2 / 0.5)
            elif key == 'q':
                self.get_logger().info("👋 종료")
                break

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotShortMove()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
