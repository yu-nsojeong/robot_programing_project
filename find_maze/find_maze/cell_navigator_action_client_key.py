import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from find_maze_interfaces.action import NavigateToCell

class NavigateClient(Node):
    def __init__(self):
        super().__init__('navigate_client')
        self._client = ActionClient(self, NavigateToCell, 'navigate_to_cell')

    def send_goal(self, row, col):
        self._client.wait_for_server()

        goal_msg = NavigateToCell.Goal()
        goal_msg.target_row = row
        goal_msg.target_col = col

        self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ 목표 거부됨')
            return

        self.get_logger().info('✅ 목표 수락됨')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'📍 현재 위치: ({fb.current_row}, {fb.current_col})')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 결과: {result.success}, 메시지: {result.message}')


def main(args=None):
    rclpy.init(args=args)
    client = NavigateClient()

    try:
        row = int(input("🎯 목표 셀 행 번호 입력 (ex. 0): "))
        col = int(input("🎯 목표 셀 열 번호 입력 (ex. 4): "))
        client.send_goal(row, col)
    except ValueError:
        print("❌ 잘못된 입력입니다. 정수를 입력하세요.")
        rclpy.shutdown()
        return

    rclpy.spin(client)
    rclpy.shutdown()



if __name__ == '__main__':
    main()
