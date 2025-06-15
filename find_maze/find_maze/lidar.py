#!/usr/bin/env python3

import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from transforms3d.euler import quat2euler
import signal
from rclpy.qos import qos_profile_sensor_data


class GlobalLidarVisualizer(Node):
    def __init__(self, plot_widget):
        super().__init__('local_lidar_visualizer')
        self.plot_widget = plot_widget
        self.scatter = self.plot_widget.plot([], [], pen=None, symbol='o', symbolSize=2)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.points = []

        self.robot_arrow = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.robot_arrow)

        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        quat = [ori.w, ori.x, ori.y, ori.z]
        _, _, yaw = quat2euler(quat)

        self.robot_x = pos.x
        self.robot_y = pos.y
        self.robot_yaw = yaw

    def scan_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        colors = []
        for x, y in zip(x_local, y_local):
            if x > 0 and abs(y) < abs(x):  # 앞
                colors.append('r')
            elif y < 0 and abs(x) < abs(y):  # 오른쪽
                colors.append('g')
            else:
                colors.append('gray')

        self.points = list(zip(x_local, y_local, colors))
        if len(self.points) > 10000:
            self.points = self.points[-10000:]

        if self.points:
            xs, ys, cs = map(list, zip(*self.points))
            self.scatter.setData(xs, ys, symbol='o', symbolSize=3, symbolBrush=cs)

        print(f"유효한 점 개수: {len(ranges)}")

        # 로봇 앞 방향 표시
        arrow_len = 0.5
        x_head = arrow_len * math.cos(0)  # 로컬 기준이므로 yaw 0
        y_head = arrow_len * math.sin(0)
        self.robot_arrow.setData([0, x_head], [0, y_head])

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiDAR Viewer - Local Frame')
        self.resize(600, 600)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.enableAutoRange()
        self.setCentralWidget(self.plot_widget)

def main():
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    node = GlobalLidarVisualizer(window.plot_widget)

    signal.signal(signal.SIGINT, lambda *_: app.quit())

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.01))
    timer.start(30)

    window.show()
    app.exec_()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
