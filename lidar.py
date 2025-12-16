#!/usr/bin/env python3
"""
AirSim Lidar to World Frame Converter with De-skewing
- 각도 기반 모션 보정으로 스캔 중 드론 회전 보상
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from collections import deque


class LidarWorldTransformer(Node):
    def __init__(self):
        super().__init__('lidar_world_transformer')

        # Odom 버퍼 (보간용)
        self.odom_buffer = deque(maxlen=100)
        self.odom_received = False

        # 최신 odom (fallback)
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # 필터링 파라미터
        self.min_distance = 2.0
        self.z_threshold = 2.0

        # 스캔 시간 (100Hz = 10ms per scan)
        self.scan_duration = 0.01  # 초

        self.get_logger().info('=== Lidar World Transformer (De-skewing) ===')
        self.get_logger().info('Angle-based motion compensation enabled')

        # AirSim odom_local 구독
        self.odom_sub = self.create_subscription(
            Odometry,
            '/airsim_node/SimpleFlight/odom_local',
            self.odom_callback,
            50
        )

        # AirSim LIDAR 구독
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/airsim_node/SimpleFlight/lidar/points/RPLIDAR_A3',
            self.lidar_callback,
            10
        )

        # 월드 좌표계 포인트클라우드 퍼블리셔
        self.world_pub = self.create_publisher(
            PointCloud2,
            '/camera/depth/points',
            10
        )

        self.count = 0

    def odom_callback(self, msg):
        """Odom을 버퍼에 저장"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        self.odom_buffer.append((timestamp, pos, quat))
        self.drone_pos = pos
        self.drone_quat = quat

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info('Odom buffer active')

    def get_odom_at_time(self, target_time):
        """특정 시간의 odom 보간"""
        if len(self.odom_buffer) < 2:
            return self.drone_pos.copy(), Rotation.from_quat(self.drone_quat).as_matrix()

        # 전후 odom 찾기
        before = None
        after = None

        for t, pos, quat in self.odom_buffer:
            if t <= target_time:
                before = (t, pos, quat)
            elif after is None:
                after = (t, pos, quat)

        if before is None:
            t, pos, quat = self.odom_buffer[0]
            return pos.copy(), Rotation.from_quat(quat).as_matrix()
        if after is None:
            t, pos, quat = self.odom_buffer[-1]
            return pos.copy(), Rotation.from_quat(quat).as_matrix()

        t1, pos1, quat1 = before
        t2, pos2, quat2 = after

        if t2 == t1:
            alpha = 0.0
        else:
            alpha = np.clip((target_time - t1) / (t2 - t1), 0.0, 1.0)

        # 위치 선형 보간
        interp_pos = pos1 + alpha * (pos2 - pos1)

        # 회전 SLERP 보간
        try:
            rotations = Rotation.from_quat([quat1, quat2])
            slerp = Slerp([0, 1], rotations)
            interp_rot = slerp(alpha).as_matrix()
        except:
            interp_rot = Rotation.from_quat(quat1).as_matrix()

        return interp_pos, interp_rot

    def lidar_callback(self, msg):
        self.count += 1

        if not self.odom_received:
            if self.count % 100 == 0:
                self.get_logger().warn('Waiting for odom...')
            return

        # 포인트클라우드 파싱
        points = self.parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        # 필터링
        xy_distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        height_diff = np.abs(points[:, 2])
        mask = (xy_distances >= self.min_distance) | ((xy_distances >= 0.5) & (height_diff > self.z_threshold))
        filtered_points = points[mask]

        if len(filtered_points) == 0:
            return

        # 스캔 타임스탬프
        scan_end_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        scan_start_time = scan_end_time - self.scan_duration

        # === De-skewing: 각 포인트별로 보간된 odom 적용 ===
        # 각 포인트의 수평 각도 계산 (-π ~ π)
        angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])

        # 각도를 0~1 진행도로 변환 (라이다가 -π에서 시작해서 π로 끝난다고 가정)
        # progress = 0: 스캔 시작, progress = 1: 스캔 끝
        progress = (angles + np.pi) / (2 * np.pi)

        # 스캔 시작/끝 odom 가져오기
        start_pos, start_rot = self.get_odom_at_time(scan_start_time)
        end_pos, end_rot = self.get_odom_at_time(scan_end_time)

        # 각 포인트 변환
        world_points = np.zeros_like(filtered_points)

        # 빠른 처리를 위해 몇 개의 구간으로 나눠서 처리
        num_segments = 8
        for seg in range(num_segments):
            seg_start = seg / num_segments
            seg_end = (seg + 1) / num_segments
            seg_mask = (progress >= seg_start) & (progress < seg_end)

            if not np.any(seg_mask):
                continue

            # 구간 중간 시점의 odom 보간
            seg_progress = (seg_start + seg_end) / 2
            seg_time = scan_start_time + seg_progress * self.scan_duration
            seg_pos, seg_rot = self.get_odom_at_time(seg_time)

            # 해당 구간 포인트 변환
            seg_points = filtered_points[seg_mask]
            world_points[seg_mask] = (seg_rot @ seg_points.T).T + seg_pos

        # 디버그 출력
        if self.count % 100 == 0:
            euler_start = Rotation.from_matrix(start_rot).as_euler('XYZ', degrees=True)
            euler_end = Rotation.from_matrix(end_rot).as_euler('XYZ', degrees=True)
            rot_diff = np.abs(euler_end - euler_start)
            self.get_logger().info(
                f'[{self.count}] De-skew: RPY diff=({rot_diff[0]:.2f}, {rot_diff[1]:.2f}, {rot_diff[2]:.2f}) deg | '
                f'Pts: {len(world_points)}'
            )

        # 발행
        world_msg = self.create_pointcloud2(world_points)
        world_msg.header.stamp = msg.header.stamp
        world_msg.header.frame_id = "world"
        self.world_pub.publish(world_msg)

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
    node = LidarWorldTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
