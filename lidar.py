#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import numpy as np
from scipy.spatial.transform import Rotation

# PX4 메시지
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude


class PointCloudFilter(Node):
    def __init__(self):
        super().__init__('pointcloud_filter')

        # 드론 위치/자세 저장 (ENU 좌표계)
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.drone_quat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.drone_rotation_matrix = np.eye(3)

        self.position_received = False
        self.attitude_received = False

        # 드론 필터링 파라미터
        self.min_distance = 2.0
        self.z_threshold = 2.0

        # Z축 스케일링 파라미터
        self.z_scale = 2.0
        self.z_scale_threshold = 2.0

        self.get_logger().info('PointCloud Filter (PX4 Direct Mode)')
        self.get_logger().info('Using PX4 odom instead of TF')

        # QoS for PX4
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # PX4 vehicle_local_position 구독
        self.px4_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_px4
        )

        # PX4 vehicle_attitude 구독
        self.px4_att_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_px4
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

    def ned_to_enu(self, x_ned, y_ned, z_ned):
        """NED 좌표를 ENU 좌표로 변환"""
        return y_ned, x_ned, -z_ned

    def vehicle_local_position_callback(self, msg):
        """PX4 위치 콜백 - NED를 ENU로 변환"""
        x_enu, y_enu, z_enu = self.ned_to_enu(msg.x, msg.y, msg.z)
        self.drone_pos = np.array([x_enu, y_enu, z_enu])

        if not self.position_received:
            self.position_received = True
            self.get_logger().info(f'PX4 Position received: X={x_enu:.2f}, Y={y_enu:.2f}, Z={z_enu:.2f}')

    def attitude_callback(self, msg):
        """PX4 자세 콜백 - FRD/NED를 ENU/FLU로 변환 (foxglove 방식)"""
        # PX4 quaternion: [w, x, y, z] - FRD body frame to NED earth frame
        q_px4 = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]

        # scipy 형식으로 변환: [x, y, z, w]
        q_frd_to_ned = [q_px4[1], q_px4[2], q_px4[3], q_px4[0]]

        # NED to ENU 변환
        q_ned_to_enu = Rotation.from_euler('ZYX', [np.pi/2, 0, np.pi], degrees=False).as_quat()

        R_frd_to_ned = Rotation.from_quat(q_frd_to_ned)
        R_ned_to_enu = Rotation.from_quat(q_ned_to_enu)

        # FRD to ENU
        R_frd_to_enu = R_ned_to_enu * R_frd_to_ned

        # FRD to FLU 변환 (x축 180도 회전)
        R_frd_to_flu = Rotation.from_euler('XYZ', [np.pi, 0, 0])

        # FLU to ENU (최종 드론 자세)
        R_flu_to_enu = R_frd_to_enu * R_frd_to_flu

        self.drone_quat = R_flu_to_enu.as_quat()  # [x, y, z, w]
        self.drone_rotation_matrix = R_flu_to_enu.as_matrix()

        if not self.attitude_received:
            self.attitude_received = True
            euler = R_flu_to_enu.as_euler('XYZ', degrees=True)
            self.get_logger().info(f'PX4 Attitude received: Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}')

    def lidar_callback(self, msg):
        self.count += 1

        # 위치/자세 수신 대기
        if not self.position_received or not self.attitude_received:
            if self.count % 50 == 0:
                self.get_logger().warn('Waiting for PX4 position/attitude...')
            return

        points = self.parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        # STEP 1: 드론 중심 필터링 (라이다 프레임)
        xy_distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        height_diff = np.abs(points[:, 2])

        mask = (xy_distances >= self.min_distance) | ((xy_distances >= 0.5) & (height_diff > self.z_threshold))
        filtered_points_lidar = points[mask]

        if len(filtered_points_lidar) == 0:
            return

        # STEP 2: 라이다 프레임 -> 월드 프레임 변환 (PX4 odom 사용)
        # 회전 적용 후 위치 이동
        world_points = (self.drone_rotation_matrix @ filtered_points_lidar.T).T + self.drone_pos

        # STEP 3: Z값 스케일링
        world_points[:, 2] = world_points[:, 2] * self.z_scale

        # 디버그 정보 출력
        if self.count % 100 == 0:
            self.get_logger().info(
                f'Drone Pos: X={self.drone_pos[0]:.2f}, Y={self.drone_pos[1]:.2f}, Z={self.drone_pos[2]:.2f} | '
                f'Points: {len(world_points)} pts (Z scaled x{self.z_scale})'
            )

        # STEP 4: 장애물 분석
        if self.count % 10 == 0:
            self.analyze_obstacles(filtered_points_lidar)

        # STEP 5: 발행
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
            warning = "!! VERY CLOSE !!"
        elif min_dist < 4.0:
            warning = "! CAUTION !"
        elif min_dist < 6.0:
            warning = "watching"

        direction = self.get_direction(closest_point)
        self.get_logger().info(
            f'Closest: {min_dist:.2f}m {direction} {warning} | '
            f'F={forward_dist:.2f}m, B={backward_dist:.2f}m, L={left_dist:.2f}m, R={right_dist:.2f}m'
        )

    def get_direction(self, point):
        x, y, z = point
        if abs(x) > abs(y):
            h_dir = "Front" if x > 0 else "Back"
        else:
            h_dir = "Left" if y > 0 else "Right"
        if abs(z) > 1.0:
            v_dir = "Up" if z > 0 else "Down"
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
