import cv2
import numpy as np
import yaml
import heapq
import numpy as np



class CellMapGenerator:
    def __init__(self, yaml_path, pgm_path, cell_size=4, grid_size=(5, 5)):
        self.yaml_path = yaml_path
        self.pgm_path = pgm_path
        self.cell_size = cell_size
        self.grid_rows, self.grid_cols = grid_size
        self.cell_map = {}

        self._load_map_info()
        self._load_image()
        self._generate_cell_map()

    # YAML 파일에서 origin과 resolution 정보 불러오기
    def _load_map_info(self):
        with open(self.yaml_path, "r") as f:
            map_info = yaml.safe_load(f)
        self.resolution = map_info["resolution"]
        self.origin_x, self.origin_y = map_info["origin"][0], map_info["origin"][1]

    # PGM 이미지 파일 로드 및 크기와 시작 위치 계산
    def _load_image(self):
        self.img = cv2.imread(self.pgm_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Image file {self.pgm_path} not found!")
        self.height, self.width = self.img.shape

        self.start_x = int(-self.origin_x / self.resolution)
        self.start_y = int(self.height + (self.origin_y / self.resolution))

    # 셀마다 상하좌우 이동 가능한지 여부 계산
    def _generate_cell_map(self):
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                cx = self.start_x + j * self.cell_size
                cy = self.start_y - i * self.cell_size

                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    continue

                cell_key = (self.grid_rows - 1 - i, j)  # 좌하단 기준
                self.cell_map[cell_key] = {
                    "up": True,
                    "down": True,
                    "left": True,
                    "right": True,
                }

                for dx in range(self.cell_size):
                    # 각 방향에 대해 해당 픽셀이 벽인지 확인
                    if (
                        0 <= cx < self.width
                        and 0 <= cy - dx < self.height
                        and self.img[cy - dx, cx] == 0
                    ):
                        self.cell_map[cell_key]["up"] = False
                    if (
                        0 <= cx < self.width
                        and 0 <= cy + dx < self.height
                        and self.img[cy + dx, cx] == 0
                    ):
                        self.cell_map[cell_key]["down"] = False
                    if (
                        0 <= cx - dx < self.width
                        and 0 <= cy < self.height
                        and self.img[cy, cx - dx] == 0
                    ):
                        self.cell_map[cell_key]["left"] = False
                    if (
                        0 <= cx + dx < self.width
                        and 0 <= cy < self.height
                        and self.img[cy, cx + dx] == 0
                    ):
                        self.cell_map[cell_key]["right"] = False

    def get_cell_map(self):
        return self.cell_map


class AStarCellPlanner:
    def __init__(self, cell_map, grid_size=(5, 5)):
        self.cell_map = cell_map
        self.grid_rows, self.grid_cols = grid_size

    # 셀 노드 클래스 (우선순위 큐용)
    class CellNode:
        def __init__(self, row, col, cost, priority):
            self.row = row
            self.col = col
            self.cost = cost
            self.priority = priority

        def __lt__(self, other):
            return self.priority < other.priority

    # 휴리스틱: 맨해튼 거리
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # A* 경로 계획 실행
    def plan(self, start, goal):
        queue = []
        heapq.heappush(queue, self.CellNode(start[0], start[1], 0, 0))
        came_from = {}
        cost_so_far = {start: 0}
        visited = set()

        while queue:
            current = heapq.heappop(queue)
            row, col = current.row, current.col

            if (row, col) == goal:
                path = []
                while (row, col) != start:
                    path.append((row, col))
                    row, col = came_from[(row, col)]
                path.append(start)
                path.reverse()
                return path

            if (row, col) in visited:
                continue
            visited.add((row, col))

            directions = {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }

            for dir_name, (dr, dc) in directions.items():
                if (row, col) not in self.cell_map:
                    continue

                if self.cell_map[(row, col)][dir_name]:  # 해당 방향으로 이동 가능하면
                    next_row, next_col = row + dr, col + dc
                    if (
                        0 <= next_row < self.grid_rows
                        and 0 <= next_col < self.grid_cols
                        and (next_row, next_col) in self.cell_map
                    ):
                        new_cost = cost_so_far[(row, col)] + 1
                        if (
                            next_row,
                            next_col,
                        ) not in cost_so_far or new_cost < cost_so_far[
                            (next_row, next_col)
                        ]:
                            cost_so_far[(next_row, next_col)] = new_cost
                            priority = new_cost + self.heuristic(
                                (next_row, next_col), goal
                            )
                            heapq.heappush(
                                queue,
                                self.CellNode(next_row, next_col, new_cost, priority),
                            )
                            came_from[(next_row, next_col)] = (row, col)

        return None  # 도달 불가

def extract_wall_info_from_scan_with_yaw(scan_msg, imu_yaw_degree):
    """
    라이다 스캔 메시지와 IMU yaw 각도(도 단위)를 사용하여 벽 정보를 추출합니다.
    """
    ranges = np.array(scan_msg.ranges)
    angle_min = np.degrees(scan_msg.angle_min) # 라디안을 도로 변환
    angle_increment = np.degrees(scan_msg.angle_increment) # 라디안을 도로 변환
    angle_max = np.degrees(scan_msg.angle_max) # 라디안을 도로 변환 (필요시)

    # direction_map은 로봇 기준의 각도가 '도' 단위로 정의되어야 합니다.
    # 예시:
    direction_map = {
        'up': 90.0,    # 로봇 정면이 0도일 때, 왼쪽 90도 (위쪽 방향)
        'down': 270.0,   # 로봇 정면이 0도일 때, 오른쪽 90도 (아래쪽 방향)
        'left': 180.0,   # 로봇 정면이 0도일 때, 뒤쪽 180도 (왼쪽 방향)
        'right': 0.0,    # 로봇 정면이 0도일 때, 정면 0도 (오른쪽 방향)
    }

    # threshold는 평균 거리 판단을 위한 임계값입니다. (단위: 미터)
    threshold = 0.3 # 예시 값. 실제 환경에 맞춰 조정 필요


    result = {}

    for cell_dir, robot_angle in direction_map.items():
        # 현재 yaw 값을 고려해 라이다 기준 실제 각도 계산
        # 모든 각도가 '도' 단위이므로 바로 뺄셈 가능
        lidar_angle = robot_angle - imu_yaw_degree

        # angle_wrap: 0~360 도 범위로 정규화
        # lidar 은 다 양수 인덱스로 계산하기.
        lidar_angle = (lidar_angle + 360.0) % 360.0


        # 라이다 index 계산
        # angle_min도 도로 변환되었으므로 그대로 사용
        index = int((lidar_angle - angle_min) / angle_increment)
        indices = range(index - 5, index + 6) # 주변 11개 인덱스

        dists = [
            ranges[i]
            for i in indices
            if 0 <= i < len(ranges) and not np.isnan(ranges[i])
        ]
        result[cell_dir] = np.mean(dists) >= threshold if dists else False


    return result
