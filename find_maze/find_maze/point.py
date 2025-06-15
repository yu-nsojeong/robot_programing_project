import cv2
import numpy as np
import yaml

# 🔧 설정 파일 경로
yaml_file = '/home/yoon/map.yaml'
pgm_file = '/home/yoon/map.pgm'
output_file = '/home/yoon/map_with_origin.png'

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
pixel_x = int(-origin_x / resolution)
pixel_y = int(height + (origin_y / resolution))  # 아래에서 위로 올라감

# 이미지 컬러로 변환 (그래야 빨간 점 찍을 수 있음)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 원 찍기
cv2.circle(img_color, (pixel_x, pixel_y), 1, (0, 0, 255), -1)  # 빨간색 점

# 결과 저장
cv2.imwrite(output_file, img_color)

print(f"✅ 저장 완료: {output_file}")
