import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from find_maze_interfaces.action import NavigateToCell

class NavigateClient(Node):
    def __init__(self):
        super().__init__('navigate_client')
        self._client = ActionClient(self, NavigateToCell, 'navigate_to_cell')

        # --- 파라미터 선언 및 값 가져오기 추가 ---
        self.declare_parameter("target_row", 0) # 기본값 0 (정수)
        self.declare_parameter("target_col", 4) # 기본값 4 (정수)

        # 파라미터 값 가져오기
        target_row = self.get_parameter("target_row").get_parameter_value().integer_value
        target_col = self.get_parameter("target_col").get_parameter_value().integer_value

        self.get_logger().info(f'🎯 목표 셀 설정: 행={target_row}, 열={target_col}')
        # --- 추가된 부분 끝 ---

        # 노드가 초기화될 때 바로 목표를 전송합니다.
        self.send_goal(target_row, target_col)


    def send_goal(self, row, col):
        self._client.wait_for_server()

        goal_msg = NavigateToCell.Goal()
        goal_msg.target_row = row
        goal_msg.target_col = col

        self.get_logger().info(f'액션 서버로 목표 전송: 행={row}, 열={col}')
        self._send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ 목표 거부됨')
            return

        self.get_logger().info('✅ 목표 수락됨')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'📍 현재 위치: ({fb.current_row}, {fb.current_col})')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 결과: {result.success}, 메시지: {result.message}')
        # 목표 달성 후 노드 종료 (선택 사항)
        # rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    client = NavigateClient()

    # rclpy.spin()은 목표 전송 및 피드백/결과 처리를 위해 계속 실행되어야 합니다.
    # input()은 제거되었으므로, 바로 스핀합니다.
    rclpy.spin(client)

    client.destroy_node() # 노드 사용이 끝나면 반드시 소멸
    rclpy.shutdown()


if __name__ == '__main__':
    main()
