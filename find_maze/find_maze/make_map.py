import cv2
import numpy as np
import yaml

# 🔧 설정 파일 경로
yaml_file = '/home/yoon/map.yaml'
pgm_file = '/home/yoon/map.pgm'
output_file = '/home/yoon/map_with_point.png'

# YAML 읽기
with open(yaml_file, 'r') as f:
    map_info = yaml.safe_load(f)

resolution = map_info['resolution']  # m/pixel
origin = map_info['origin']          # [x, y, yaw]
negate = map_info.get('negate', 0)

# 이미지 읽기 (grayscale 유지)
img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)

# 이미지가 None이면 에러
if img is None:
    raise FileNotFoundError(f"Image file {pgm_file} not found!")

height, width = img.shape

# origin 좌표를 이미지 픽셀로 변환
# origin은 좌하단이 기준이고, 이미지 좌표는 좌상단이 기준이므로 Y축 뒤집어야 함
origin_x, origin_y = origin[0], origin[1]

# origin을 픽셀로 변환
start_x = int(-origin_x / resolution)
start_y = int(height + (origin_y / resolution))  # 아래에서 위로 올라감


length = 4
directions = ['up', 'down', 'left', 'right']
cell_map = {}

for i in range(5):  # Y축 위 방향 (행: 0~4)
    for j in range(5):  # X축 오른쪽 방향 (열: 0~4)
        cx = start_x + j*4
        cy = start_y - i*4  # OpenCV는 y가 아래로 갈수록 +니까 위로 가려면 -해야 함

        if not (0 <= cx < width and 0 <= cy < height):
            continue


        cell_key = (4-i, j)  # 좌하단 기준 0~4 인덱스
        cell_map[cell_key] = {'up': True, 'down': True, 'left': True, 'right': True}

        for dx in range(4):
            # 위 검사
            y_up = cy - dx
            x_up = cx
            if 0 <= x_up < width and 0 <= y_up < height and img[y_up, x_up] == 0:
                cell_map[cell_key]['up'] = False

            # 아래 검사
            y_down = cy + dx
            x_down = cx
            if 0 <= x_down < width and 0 <= y_down < height and img[y_down, x_down] == 0:
                cell_map[cell_key]['down'] = False

            # 왼쪽 검사
            x_left = cx - dx
            y_left = cy
            if 0 <= x_left < width and 0 <= y_left < height and img[y_left, x_left] == 0:
                cell_map[cell_key]['left'] = False

            # 오른쪽 검사
            x_right = cx + dx
            y_right = cy
            if 0 <= x_right < width and 0 <= y_right < height and img[y_right, x_right] == 0:
              cell_map[cell_key]['right'] = False


        #cv2.circle(img, (cx, cy), 1, (0, 0, 0), -1)  # 검정색 점

print("\n🧭 이동 가능 방향 (5x5 셀 기준):")
for row in range(5):
    for col in range(5):
        print(f"셀({row},{col}): {cell_map.get((row, col), '밖임')}")


import heapq

class CellNode:
    def __init__(self, row, col, cost, priority):
        self.row = row
        self.col = col
        self.cost = cost
        self.priority = priority
    def __lt__(self, other):
        return self.priority < other.priority

def heuristic(a, b):
    # 맨해튼 거리
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_on_cells(cell_map, start=(4, 0), goal=(0, 4)):
    queue = []
    heapq.heappush(queue, CellNode(start[0], start[1], 0, 0))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while queue:
        current = heapq.heappop(queue)
        row, col = current.row, current.col

        if (row, col) == goal:
            # 경로 추적
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
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        for dir_name, (dr, dc) in directions.items():
            if (row, col) not in cell_map:
                continue

            if cell_map[(row, col)][dir_name]:  # 이동 가능하면
                next_row, next_col = row + dr, col + dc
                if 0 <= next_row < 5 and 0 <= next_col < 5 and (next_row, next_col) in cell_map:
                    new_cost = cost_so_far[(row, col)] + 1
                    if (next_row, next_col) not in cost_so_far or new_cost < cost_so_far[(next_row, next_col)]:
                        cost_so_far[(next_row, next_col)] = new_cost
                        priority = new_cost + heuristic((next_row, next_col), goal)
                        heapq.heappush(queue, CellNode(next_row, next_col, new_cost, priority))
                        came_from[(next_row, next_col)] = (row, col)

    return None  # 경로 없음

# 🚀 A* 실행
path = a_star_on_cells(cell_map)

print("\n🟢 A* 경로 결과:")
if path:
    for step in path:
        print(f"셀(row={step[0]}, col={step[1]})")
else:
    print("❌ 도달 불가")


import numpy as np

def extract_wall_info_from_scan(scan_msg, threshold=0.3):
    """
    scan_msg: sensor_msgs.msg.LaserScan
    threshold: 벽으로 간주할 거리(m) 이하면 벽으로 판단
    """
    ranges = np.array(scan_msg.ranges)
    angle_min = scan_msg.angle_min
    angle_increment = scan_msg.angle_increment

    # 각 방향별 라디안 기준
    direction_angles = {
        'front': 0.0,
        'right': -np.pi / 2,
        'back': np.pi,
        'left': np.pi / 2
    }

    direction_map = {
        'up': 'front',
        'down': 'back',
        'left': 'left',
        'right': 'right'
    }

    result = {}

    for key, direction in direction_map.items():
        target_angle = direction_angles[direction]
        index = int((target_angle - angle_min) / angle_increment)
        indices = range(index - 5, index + 6)  # ±5도 범위 평균

        dists = []
        for i in indices:
            if 0 <= i < len(ranges) and not np.isnan(ranges[i]):
                dists.append(ranges[i])

        if not dists:
            result[key] = False  # 데이터 없음 → 벽 아님으로 간주
        else:
            mean_dist = np.mean(dists)
            result[key] = mean_dist <= threshold  # 가까우면 벽

    return result


# 이미지 컬러로 변환 (그래야 빨간 점 찍을 수 있음)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 원 찍기
cv2.circle(img_color, (start_x, start_y), 1, (0, 0, 255), -1)  # 빨간색 점

cv2.imshow("image",img_color)

# 결과 저장
cv2.imwrite(output_file, img_color)

print(f"✅ 저장 완료: {output_file}")
