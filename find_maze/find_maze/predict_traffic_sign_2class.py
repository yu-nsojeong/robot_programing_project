import cv2
import numpy as np
from tensorflow.keras.models import load_model
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32   # ✅ String → Int32로 변경
import threading
import os
from ament_index_python.packages import get_package_share_directory

# =======================
# ROS2 노드 클래스 정의
# =======================
class TrafficSignPublisher(Node):
    def __init__(self):
        super().__init__('traffic_sign_publisher')
        self.publisher_ = self.create_publisher(Int32, 'traffic_sign_result', 10)  # ✅ Int32 사용

    def publish_result(self, value):
        msg = Int32()
        msg.data = value
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

# =======================
# 메인 실행 함수
# =======================
def main():
    rclpy.init()
    node = TrafficSignPublisher()

    def ros_spin():
        rclpy.spin(node)
    thread = threading.Thread(target=ros_spin, daemon=True)
    thread.start()

    # 현재 스크립트 파일의 디렉토리 경로를 가져옵니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))


    #model = load_model("/home/yoon/turtlebot3_ws/src/find_maze/find_maze/model/gtsrb_model_2class.keras")

    # # 모델 경로 설정 (ROS 2 패키지 내 model 디렉토리)
    package_share_directory = get_package_share_directory("find_maze")
    model_path = os.path.join(package_share_directory, "model", "gtsrb_model_2class.keras")
    # 모델 로드
    model = load_model(model_path)

    print("로드 완료")
    classes = {
        0: 'Speed limit (30km/h)',
        1: 'Speed limit (70km/h)'
    }

    IMG_SIZE = 32

    cap = cv2.VideoCapture("udp://@172.100.2.21:5000?overrun_nonfatal=1")
    if not cap.isOpened():
        print("웹캠 스트림을 열 수 없습니다.")
        return

    print("실시간 교통 표지판 인식 시작 (q를 눌러 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패. 스트림이 끊어졌거나 종료되었습니다.")
            break

        h, w, _ = frame.shape
        center_crop = frame[h//2-64:h//2+64, w//2-64:w//2+64]
        if center_crop.shape[0] != 128 or center_crop.shape[1] != 128:
            continue

        img = cv2.resize(center_crop, (IMG_SIZE, IMG_SIZE))
        input_img = np.expand_dims(img / 255.0, axis=0)

        predictions = model.predict(input_img)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)
        label = classes.get(class_id, f"Unknown ({class_id})")

        # 화면에 출력
        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Webcam - Traffic Sign Detection", frame)

        # 결과값 정수로 변환
        if label == 'Speed limit (30km/h)':
            numeric_value = 30
        else:
            numeric_value = 70

        # 퍼블리시
        node.publish_result(numeric_value)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
