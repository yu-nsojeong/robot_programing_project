# #!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from transforms3d.euler import quat2euler
import math

class ImuYawReader(Node):
    def __init__(self):
        super().__init__('imu_yaw_reader')

        # IMU 서브스크라이버
        self.subscription = self.create_subscription(
            Imu,
            '/imu',  # 필요 시 수정
            self.imu_callback,
            10
        )

        # yaw(rad) 퍼블리셔
        self.yaw_pub = self.create_publisher(Float32, '/imu_yaw', 10)

        # yaw(deg) 퍼블리셔
        self.yaw_deg_pub = self.create_publisher(Float32, '/imu_yaw_deg', 10)

        # 초기 yaw 값 저장 변수
        self.initial_yaw_offset = None
        self.calibration_duration_sec = 5  # 캘리브레이션 지속 시간 (초)
        self.start_time = self.get_clock().now()

    def imu_callback(self, msg):
        q = msg.orientation
        quat_t3d = [q.w, q.x, q.y, q.z]

        # `transforms3d`의 `quat2euler`는 (roll, pitch, yaw) 순서로 반환합니다.
        roll, pitch, yaw = quat2euler(quat_t3d)

        # 캘리브레이션 단계
        if self.initial_yaw_offset is None:
            # 캘리브레이션 지속 시간 동안 초기 yaw 값을 평균내어 저장할 수도 있습니다.
            # 여기서는 단순히 첫 값을 사용합니다.
            if (self.get_clock().now() - self.start_time).nanoseconds / 1e9 < self.calibration_duration_sec:
                # 캘리브레이션 시간 동안 계속해서 initial_yaw_offset을 업데이트합니다.
                # 이렇게 하면 캘리브레이션 시간 동안의 마지막 값이 초기 오프셋으로 설정됩니다.
                # 더 견고하게 하려면 여러 값을 평균내는 로직을 추가할 수 있습니다.
                self.initial_yaw_offset = yaw
                self.get_logger().info(f"Calibrating initial yaw... Current yaw: {yaw:.4f} rad")
                return # 캘리브레이션 중에는 0으로 퍼블리시하지 않습니다.

            else:
                self.get_logger().info(f"Calibration complete. Initial yaw offset set to: {self.initial_yaw_offset:.4f} rad")
                # 캘리브레이션이 완료되었으니 현재 yaw 값을 오프셋으로 설정하고 다음부터는 상대적인 값을 사용합니다.

        # 캘리브레이션된 yaw 값 계산
        calibrated_yaw = yaw - self.initial_yaw_offset

        # -pi ~ pi 범위로 정규화
        # 이 부분은 옵션이지만, yaw 값이 계속 증가하거나 감소하는 것을 방지합니다.
        calibrated_yaw = math.atan2(math.sin(calibrated_yaw), math.cos(calibrated_yaw))


        yaw_deg = math.degrees(calibrated_yaw)
        print(f"Calibrated Yaw: {calibrated_yaw:.4f} rad, {yaw_deg:.2f} deg")

        # rad 퍼블리시
        yaw_msg = Float32()
        yaw_msg.data = calibrated_yaw
        self.yaw_pub.publish(yaw_msg)

        # deg 퍼블리시
        yaw_deg_msg = Float32()
        yaw_deg_msg.data = yaw_deg
        self.yaw_deg_pub.publish(yaw_deg_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuYawReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Imu
# from std_msgs.msg import Float32
# from transforms3d.euler import quat2euler
# import math

# class ImuYawReader(Node):
#     def __init__(self):
#         super().__init__('imu_yaw_reader')

#         # IMU 서브스크라이버
#         self.subscription = self.create_subscription(
#             Imu,
#             '/imu',  # 필요 시 수정
#             self.imu_callback,
#             10
#         )

#         # yaw(rad) 퍼블리셔
#         self.yaw_pub = self.create_publisher(Float32, '/imu_yaw', 10)

#         # yaw(deg) 퍼블리셔
#         self.yaw_deg_pub = self.create_publisher(Float32, '/imu_yaw_deg', 10)

#     def imu_callback(self, msg):
#         q = msg.orientation
#         quat_t3d = [q.w, q.x, q.y, q.z]
#         roll, pitch, yaw = quat2euler(quat_t3d)

#         yaw_deg = math.degrees(yaw)
#         #self.get_logger().info(f"Yaw: {yaw:.4f} rad, {yaw_deg:.2f} deg")
#         print(f"Yaw: {yaw:.4f} rad, {yaw_deg:.2f} deg")

#         # rad 퍼블리시
#         yaw_msg = Float32()
#         yaw_msg.data = yaw
#         self.yaw_pub.publish(yaw_msg)

#         # deg 퍼블리시
#         yaw_deg_msg = Float32()
#         yaw_deg_msg.data = yaw_deg
#         self.yaw_deg_pub.publish(yaw_deg_msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ImuYawReader()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
