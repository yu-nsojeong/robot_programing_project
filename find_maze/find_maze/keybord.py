import rclpy
from rclpy.node import Node
from find_maze_interfaces.srv import Move
import sys, tty, termios

class TurtleBotKeyboardClient(Node):
    def __init__(self):
        super().__init__('turtlebot_keyboard_client')
        self.cli = self.create_client(Move, 'move_robot')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서비스 연결 대기 중...')
        self.get_logger().info("🕹 방향키 조작: ↑ 30cm / ↓ 180도 후 30cm / ← 90도 회전 / → 90도 회전")

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch1 = sys.stdin.read(1)
            if ch1 == '\x1b':
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

    def send_request(self, degree, distance):
        req = Move.Request()
        req.degree = degree
        req.distance = distance
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info(f"✅ 이동 완료: {future.result().message}")
        else:
            self.get_logger().error("❌ 서비스 호출 실패")

    def run(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'UP':
                self.get_logger().info("⬆️  전진 40cm")
                self.send_request(degree=0.0, distance=0.4)
            elif key == 'DOWN':
                self.get_logger().info("⬇️ 후진 40cm")
                self.send_request(degree=0.0, distance=-0.4)
            elif key == 'LEFT':
                self.get_logger().info("⬅️  왼쪽 90도 회전")
                self.send_request(degree=90.0, distance=0.0)
            elif key == 'RIGHT':
                self.get_logger().info("➡️  오른쪽 90도 회전")
                self.send_request(degree=-90.0, distance=0.0)
            elif key == 'q':
                self.get_logger().info("종료")
                break

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotKeyboardClient()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
