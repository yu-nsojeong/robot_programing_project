from setuptools import find_packages, setup
from ament_index_python.packages import get_package_share_directory
import os
from glob import glob

package_name = 'find_maze'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('find_maze','launch', '*launch.[py]*'))),
        (os.path.join('share', package_name, 'map'), glob(os.path.join('find_maze', 'map', '*.yaml'))),
        (os.path.join('share', package_name, 'map'), glob(os.path.join('find_maze', 'map', '*.pgm'))),
        (os.path.join('share', package_name, 'model'), glob(os.path.join('find_maze', 'model', '*.keras'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoon',
    maintainer_email='chaely02@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'key = find_maze.keybord:main',
            'imu = find_maze.imucallback:main',
            'lidar = find_maze.lidar:main',
            'right = find_maze.right_hand_wall_following:main',
            'teleop = find_maze.turtlebot_control:main',
            'fira = find_maze.fira_maze:main',
            'ac = find_maze.cell_navigator_action_client:main',
            'kac = find_maze.cell_navigator_action_client_key:main',
            'as = find_maze.cell_navigator_action_server:main',
            'vision = find_maze.predict_traffic_sign_2class:main',
            'back = find_maze.right_wall_following_back:main',
        ],
    },
)
