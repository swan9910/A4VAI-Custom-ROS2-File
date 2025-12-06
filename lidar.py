#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from tf2_ros import TransformListener, Buffer
from scipy.spatial.transform import Rotation

class PointCloudFilter(Node):
    def __init__(self):
        super().__init__('pointcloud_filter')

        # TF Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # AirSim 초기 위치 오프셋
        self.initial_position_set = False
        self.initial_airsim_x = -27.5
        self.initial_airsim_y = 27.5
        self.initial_airsim_z = 0.0

        # 드론 위치 저장
        self.drone_pos = np.array([0.0, 0.0, 0.0])

        # 드론 필터링 파라미터
        self.min_distance = 2.0
        self.z_threshold = 2.0

        # Z축 스케일링 파라미터 (새 포인트 생성 없이 값만 2배)
        self.z_scale = 2.0           # z값을 2배로 스케일링
        self.z_scale_threshold = 2.0  # z > 2m인 포인트만 스케일링

        self.get_logger().info(f'🔧 PointCloud Filter with Z-Scaling (NO new points)')
        self.get_logger().info(f'   z > {self.z_scale_threshold}m인 포인트의 z값을 {self.z_scale}배로 스케일링')

        # AirSim odom_local 구독
        self.airsim_odom_sub = self.create_subscription(
            Odometry,
            '/airsim_node/SimpleFlight/odom_local',
            self.airsim_odom_callback,
            10
        )

        # ego_odom 구독
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_odom',
            self.odom_callback,
            10
        )

        # LIDAR 구독
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3',
            self.lidar_callback,
            10
        )

        # 퍼블리셔
        self.filtered_pub = self.create_publisher(
            PointCloud2,
            '/camera/depth/points',
            10
        )

        self.count = 0

    def airsim_odom_callback(self, msg):
        if not self.initial_position_set:
            self.initial_airsim_x = -27.5
            self.initial_airsim_y = 27.5
            self.initial_airsim_z = 0.0
            self.initial_position_set = True
            self.get_logger().info(
                f'📍 Initial AirSim Position: '
                f'X={self.initial_airsim_x:.2f}, Y={self.initial_airsim_y:.2f}, Z={self.initial_airsim_z:.2f}'
            )

    def odom_callback(self, msg):
        self.drone_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

    def lidar_callback(self, msg):
        self.count += 1

        points = self.parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        # STEP 1: 드론 중심 필터링
        xy_distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        height_diff = np.abs(points[:, 2])

        mask = (xy_distances >= self.min_distance) | ((xy_distances >= 0.5) & (height_diff > self.z_threshold))
        filtered_points_lidar = points[mask]

        if len(filtered_points_lidar) == 0:
            return

        # STEP 2: TF 변환
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            rotation_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]

            rotation_matrix = Rotation.from_quat(rotation_quat).as_matrix()
            world_points_absolute = (rotation_matrix @ filtered_points_lidar.T).T + translation

            # STEP 3: AirSim 초기 위치 오프셋 적용
            if self.initial_position_set:
                offset = np.array([
                    self.initial_airsim_x,
                    self.initial_airsim_y,
                    self.initial_airsim_z
                ])
                world_points = world_points_absolute - offset
            else:
                if self.count % 50 == 0:
                    self.get_logger().warn('Waiting for initial AirSim position...')
                return

            # ⭐ STEP 4: Z값 스케일링 (새 포인트 생성 없이!)
            # z > 2m인 포인트의 z값만 2배로 스케일링
            # 모든 포인트의 z값을 2배로 (조건 없이)
            scaled_count = len(world_points)

            if True:  # 항상 실행
                # z > 2m인 포인트의 z값을 2배로
                world_points[:, 2] = world_points[:, 2] * self.z_scale  # 모든 z값 2배

            # 디버그 정보 출력
            if self.count % 100 == 0:
                self.get_logger().info(
                    f'🔍 TF: X={translation[0]:.2f}, Y={translation[1]:.2f}, Z={translation[2]:.2f}\n'
                    f'   Drone: X={self.drone_pos[0]:.2f}, Y={self.drone_pos[1]:.2f}, Z={self.drone_pos[2]:.2f}\n'
                    f'   📊 Total: {len(world_points)} pts, Z-scaled: {scaled_count} pts (z>{self.z_scale_threshold}m → {self.z_scale}x)'
                )

        except Exception as e:
            if self.count % 50 == 0:
                self.get_logger().warn(f'TF lookup failed: {e}')
            return

        # STEP 5: 장애물 분석
        if self.count % 10 == 0:
            self.analyze_obstacles(filtered_points_lidar)

        # STEP 6: 발행 (포인트 수는 원본과 동일!)
        filtered_msg = self.create_pointcloud2(world_points)
        filtered_msg.header.stamp = msg.header.stamp
        filtered_msg.header.frame_id = "world"
        self.filtered_pub.publish(filtered_msg)

    def analyze_obstacles(self, points):
        if len(points) == 0:
            return
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        closest_point = points[min_idx]

        forward_mask = points[:, 0] > 0
        backward_mask = points[:, 0] < 0
        left_mask = points[:, 1] > 0
        right_mask = points[:, 1] < 0

        forward_dist = distances[forward_mask].min() if forward_mask.any() else 999.0
        backward_dist = distances[backward_mask].min() if backward_mask.any() else 999.0
        left_dist = distances[left_mask].min() if left_mask.any() else 999.0
        right_dist = distances[right_mask].min() if right_mask.any() else 999.0

        warning = ""
        if min_dist < 2.5:
            warning = "⚠️  매우 가까움!"
        elif min_dist < 4.0:
            warning = "⚡ 주의!"
        elif min_dist < 6.0:
            warning = "👀 관찰 중"

        direction = self.get_direction(closest_point)
        self.get_logger().info(
            f'🎯 최근접: {min_dist:.2f}m {direction} {warning}\n'
            f'   🧭 전방={forward_dist:.2f}m, 후방={backward_dist:.2f}m, '
            f'좌={left_dist:.2f}m, 우={right_dist:.2f}m'
        )

    def get_direction(self, point):
        x, y, z = point
        if abs(x) > abs(y):
            h_dir = "전방" if x > 0 else "후방"
        else:
            h_dir = "좌측" if y > 0 else "우측"
        if abs(z) > 1.0:
            v_dir = "위" if z > 0 else "아래"
            return f"({h_dir}-{v_dir})"
        else:
            return f"({h_dir})"

    def parse_pointcloud2(self, cloud_msg):
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        points_struct = np.frombuffer(cloud_msg.data, dtype=dtype)
        points = np.column_stack((points_struct['x'], points_struct['y'], points_struct['z']))
        valid_mask = ~np.isnan(points).any(axis=1)
        return points[valid_mask] if np.any(valid_mask) else None

    def create_pointcloud2(self, points):
        msg = PointCloud2()
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.fields = [
            self.create_field('x', 0, 7, 1),
            self.create_field('y', 4, 7, 1),
            self.create_field('z', 8, 7, 1)
        ]
        msg.data = points.astype(np.float32).tobytes()
        return msg

    def create_field(self, name, offset, datatype, count):
        from sensor_msgs.msg import PointField
        field = PointField()
        field.name = name
        field.offset = offset
        field.datatype = datatype
        field.count = count
        return field

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
