import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        self.sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        self.range_front = [0.0]*10
        self.range_right = []
        self.range_left = []
        self.min_front = 0.0
        self.min_right = 0.0
        self.min_left = 0.0

        self.timer = self.create_timer(0.1, self.main_loop)  # 10Hz
        self.command = Twist()
        self.near_wall = 0
        self.ready = False

        self.get_logger().info('WallFollower node initialized. Turning right...')
        time.sleep(1.0)
        self.command.angular.z = -0.5
        self.command.linear.x = 0.1
        self.cmd_pub.publish(self.command)
        time.sleep(2.0)

    def scan_callback(self, msg):
        ranges = msg.ranges
        self.range_front = list(ranges[5:0:-1]) + list(ranges[-1:-5:-1])
        self.range_right = ranges[300:345]
        self.range_left = ranges[60:15:-1]

        self.min_front = min(self.range_front)
        self.min_right = min(self.range_right)
        self.min_left = min(self.range_left)
        self.ready = True

    def main_loop(self):
        if not self.ready:
            return

        if self.near_wall == 0:
            if self.min_front > 0.2 and self.min_right > 0.2 and self.min_left > 0.2:
                self.command.angular.z = -0.1
                self.command.linear.x = 0.05
            elif self.min_left < 0.2:
                self.near_wall = 1
            else:
                self.command.angular.z = -0.25
                self.command.linear.x = 0.0
        else:
            if self.min_front > 0.2:
                if self.min_left < 0.12:
                    self.command.angular.z = -0.5
                    self.command.linear.x = -0.05
                elif self.min_left > 0.15:
                    self.command.angular.z = 0.5
                    self.command.linear.x = 0.05
                else:
                    self.command.angular.z = -0.5
                    self.command.linear.x = 0.05
            else:
                self.command.angular.z = -0.5
                self.command.linear.x = 0.0

        self.cmd_pub.publish(self.command)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
