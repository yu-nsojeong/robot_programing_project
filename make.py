import cv2
import numpy as np
import yaml
import heapq
import random
import matplotlib.pyplot as plt

# A* 알고리즘용 노드 클래스
class Node:
    def __init__(self, x, y, cost, priority):
        self.x = x
        self.y = y
        self.cost = cost
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

# A* 알고리즘 (벽 가까운 곳 피하는 버전)
def astar(map_data, start, goal, dist_map, wall_weight=10.0):
    height, width = map_data.shape
    visited = np.zeros_like(map_data, dtype=bool)
    came_from = {}
    cost_so_far = {start: 0}
    queue = []
    heapq.heappush(queue, Node(start[0], start[1], 0, 0))

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),     # 상하좌우
        (-1, -1), (-1, 1), (1, -1), (1, 1)    # 대각선
    ]

    while queue:
        current = heapq.heappop(queue)
        cx, cy = current.x, current.y

        if (cx, cy) == goal:
            path = []
            while (cx, cy) != start:
                path.append((cx, cy))
                cx, cy = came_from[(cx, cy)]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                if visited[ny][nx] or map_data[ny][nx] < 250:
                    continue

                move_cost = 1.0 if dx == 0 or dy == 0 else 1.4
                penalty = (1.0 / (dist_map[ny][nx] + 1e-5)) * wall_weight
                new_cost = cost_so_far[(cx, cy)] + move_cost + penalty

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    heuristic = abs(goal[0] - nx) + abs(goal[1] - ny)
                    priority = new_cost + heuristic
                    heapq.heappush(queue, Node(nx, ny, new_cost, priority))
                    came_from[(nx, ny)] = (cx, cy)

        visited[cy][cx] = True

    return None

# 이동 가능한 랜덤 위치 2개 선택
def get_random_free_positions(img, count=2):
    free_positions = list(zip(*np.where(img >= 254)))  # (y, x)
    if len(free_positions) < count:
        raise ValueError("이동 가능한 픽셀이 충분하지 않음!")
    selected = random.sample(free_positions, count)
    return [(x, y) for y, x in selected]  # (x, y) 형태로 반환

# 경로 시각화
def draw_path_on_map(img, path):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y) in path:
        vis[y, x] = (0, 0, 255)  # 빨간색
    return vis

# === 메인 실행 ===

# 파일 경로 설정
pgm_file = '/home/yoon/turtlebot3_ws/.pgm'
yaml_file = '/home/yoon/turtlebot3_ws/.yaml'

# YAML에서 해상도, origin 등 로딩
with open(yaml_file, 'r') as f:
    map_metadata = yaml.safe_load(f)

resolution = map_metadata['resolution']
origin = map_metadata['origin']

# 이미지 로딩
img = cv2.imread(pgm_file, cv2.IMREAD_UNCHANGED)
height, width = img.shape
print(f"📏 맵 크기: {width} x {height}")

# 거리 맵 계산
binary_map = np.uint8(img >= 254)  # 자유 공간만 1, 나머지 0
dist_map = cv2.distanceTransform(binary_map, distanceType=cv2.DIST_L2, maskSize=5)

# 거리 맵 시각화
plt.figure(figsize=(6, 5))
plt.imshow(dist_map, cmap='inferno')
plt.title("🟠 벽에서 떨어진 거리 맵")
plt.colorbar(label='Distance from obstacle')
plt.axis("off")
plt.show()

# 픽셀 분포 출력
unique, counts = np.unique(img, return_counts=True)
pixel_stats = dict(zip(unique, counts))
print("📊 픽셀 값 분포:")
for val, count in pixel_stats.items():
    print(f"값 {val}: {count}개")

# 랜덤한 이동 가능한 위치 2개 선택
start, goal = get_random_free_positions(img, 2)
print(f"🚩 Start: {start}, 🏁 Goal: {goal}")

# 경로 계산 (거리 맵 반영)
path = astar(img, start, goal, dist_map)

# 시각화
if path:
    print(f"✅ 경로 길이: {len(path)}")
    vis = draw_path_on_map(img, path)
    plt.imshow(vis)
    plt.title("경로 시각화 (빨간색)")
    plt.axis("off")
    plt.show()
else:
    print("❌ 경로를 찾을 수 없습니다.")
