
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from quadrotor_msgs.msg import PositionCommand
import math
import numpy as np

class EgoPlannerPX4Controller(Node):

    def __init__(self) -> None:
        super().__init__('ego_planner_px4_controller')


        self.initial_position_set = False
        self.initial_airsim_x = 0.0
        self.initial_airsim_y = 0.0
        self.initial_airsim_z = 0.0

        # QoS 프로파일 설정
        px4_pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        px4_sub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # PX4 퍼블리셔
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_pub_qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_pub_qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_pub_qos_profile)

        # PX4 서브스크라이버
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, px4_sub_qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, px4_sub_qos_profile)

        # AirSim 서브스크라이버
        self.airsim_odom_sub = self.create_subscription(
            Odometry, '/airsim_node/SimpleFlight/odom_local', self.airsim_odom_callback, 10)

        # EGO-Planner 서브스크라이버
        self.ego_planner_sub = self.create_subscription(
            PositionCommand,
            '/planning/pos_cmd',
            self.ego_planner_callback,
            10
        )

        # ⭐ EGO-Planner용 퍼블리셔 (nav_msgs/Odometry)
        self.odom_pub = self.create_publisher(Odometry, '/ego_odom', 10)           # FSM용
        self.odom_grid_pub = self.create_publisher(Odometry, '/ego_odom_grid', 10) # grid_map용
        self.pose_pub = self.create_publisher(PoseStamped, '/ego_pose', 10)        # grid_map/pose용

        # 상태 변수
        self.offboard_setpoint_counter = 0
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.airsim_odom = Odometry()

        # 드론 상태 관리
        self.STATE_INIT = 0
        self.STATE_ARMING = 1
        self.STATE_TAKEOFF = 2
        self.STATE_PLANNER_READY = 3

        self.current_state = self.STATE_INIT
        self.takeoff_height = -5.0
        self.tolerance = 0.15

        # 제어 파라미터
        self.target_altitude = -5.0
        self.max_velocity = 4.0  # 장애물 회피를 위해 속도 제한 (4.0 → 0.5)
        self.min_cmd_threshold = 0.02

        # 현재 드론 상태 (ENU 좌표계)
        self.current_position = [0.0, 0.0, 0.0]
        self.current_velocity = [0.0, 0.0, 0.0]

        # EGO-Planner 명령 타임아웃
        self.last_planner_cmd_time = self.get_clock().now()
        self.planner_cmd_timeout = 0.5

        # 데이터 유효성 플래그
        self.odometry_data_valid = False
        self.airsim_odom_valid = False
        self.pointcloud_received = False
        self.pointcloud_count = 0

        # 타이머 설정 (20Hz 제어)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('🚁 EGO-Planner + PX4 Controller initialized')
#        self.get_logger().info(f'⚙️  AirSim Offset: X={self.airsim_offset_x}, Y={self.airsim_offset_y}, Z={self.airsim_offset_z}')

    def airsim_odom_callback(self, msg):
        """AirSim Odometry → EGO-Planner Odometry (오프셋 보정 포함)"""
        self.airsim_odom = msg
        self.airsim_odom_valid = True

        # ⭐ 초기 위치 자동 캘리브레이션 (첫 번째 메시지에서)
        if not self.initial_position_set:
            self.initial_airsim_x = -27.5
            self.initial_airsim_y = 27.5
            self.initial_airsim_z = 0.0
            self.initial_position_set = True
            self.get_logger().info(f'📍 Initial AirSim Position: X={self.initial_airsim_x:.2f}, Y={self.initial_airsim_y:.2f}, Z={self.initial_airsim_z:.2f}')

        # ⭐ 오프셋 보정: AirSim 좌표 → EGO-Planner 좌표 (원점 기준)
        corrected_x = msg.pose.pose.position.x - self.initial_airsim_x
        corrected_y = msg.pose.pose.position.y - self.initial_airsim_y
        corrected_z = msg.pose.pose.position.z - self.initial_airsim_z

        # 1. nav_msgs/Odometry 발행 (FSM & grid_map용)
        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "base_link"

        # ⭐ 보정된 위치 설정
        odom_msg.pose.pose.position.x = corrected_x
        odom_msg.pose.pose.position.y = corrected_y
        odom_msg.pose.pose.position.z = corrected_z

        # ⭐ AirSim → ROS2 좌표계 변환: Orientation 90도 보정
        # AirSim은 90도 회전된 좌표계 사용 (QGC 0도 = AirSim 90도)
        airsim_quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        # Z축 -90도 회전 quaternion (AirSim → ROS2 ENU)
        rotation_quat = [0.0, 0.0, -0.7703311, 0.6376440]  # -100.77도

        # Quaternion 곱셈으로 보정
        def quat_multiply(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return [
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ]

        corrected_quat = quat_multiply(airsim_quat, rotation_quat)

        odom_msg.pose.pose.orientation.x = corrected_quat[0]
        odom_msg.pose.pose.orientation.y = corrected_quat[1]
        odom_msg.pose.pose.orientation.z = corrected_quat[2]
        odom_msg.pose.pose.orientation.w = corrected_quat[3]
        odom_msg.pose.covariance = msg.pose.covariance

        # Twist(속도) 그대로 유지
        odom_msg.twist = msg.twist

        self.odom_pub.publish(odom_msg)
        self.odom_grid_pub.publish(odom_msg)

        # 2. PoseStamped 발행 (grid_map/pose용)
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = corrected_x
        pose_msg.pose.position.y = corrected_y
        pose_msg.pose.position.z = corrected_z
        pose_msg.pose.orientation.x = corrected_quat[0]
        pose_msg.pose.orientation.y = corrected_quat[1]
        pose_msg.pose.orientation.z = corrected_quat[2]
        pose_msg.pose.orientation.w = corrected_quat[3]

        self.pose_pub.publish(pose_msg)

    def vehicle_odometry_callback(self, vehicle_odometry):
        """PX4 드론 위치 (NED → ENU 변환)"""
        self.vehicle_odometry = vehicle_odometry

        self.current_position = [
            float(vehicle_odometry.position[1]),
            float(vehicle_odometry.position[0]),
            float(-vehicle_odometry.position[2])
        ]

        self.current_velocity = [
            float(vehicle_odometry.velocity[1]),
            float(vehicle_odometry.velocity[0]),
            float(-vehicle_odometry.velocity[2])
        ]

        self.odometry_data_valid = True

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def ego_planner_callback(self, cmd_msg):
        if self.current_state != self.STATE_PLANNER_READY:
            return

        self.last_planner_cmd_time = self.get_clock().now()

        # ⭐ EGO-Planner의 position & velocity & acceleration 명령 사용
        pos_x_enu = cmd_msg.position.x
        pos_y_enu = cmd_msg.position.y
        pos_z_enu = cmd_msg.position.z

        vel_x_enu = cmd_msg.velocity.x
        vel_y_enu = cmd_msg.velocity.y
        vel_z_enu = cmd_msg.velocity.z

        acc_x_enu = cmd_msg.acceleration.x
        acc_y_enu = cmd_msg.acceleration.y
        acc_z_enu = cmd_msg.acceleration.z

        # 속도 제한 (안전)
        vel_x_enu = np.clip(vel_x_enu, -self.max_velocity, self.max_velocity)
        vel_y_enu = np.clip(vel_y_enu, -self.max_velocity, self.max_velocity)
        vel_z_enu = np.clip(vel_z_enu, -self.max_velocity, self.max_velocity)

        # ENU → PX4 NED 변환: x↔y 교환, z 부호 반전
        # Position, Velocity, Acceleration 모두 사용 (feedforward control)
        msg = TrajectorySetpoint()
        msg.position = [float(pos_y_enu), float(pos_x_enu), float(-pos_z_enu)]
        msg.velocity = [float(vel_y_enu), float(vel_x_enu), float(-vel_z_enu)]
        msg.acceleration = [float(acc_y_enu), float(acc_x_enu), float(-acc_z_enu)]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.trajectory_setpoint_publisher.publish(msg)

        vel_mag = math.sqrt(vel_x_enu**2 + vel_y_enu**2 + vel_z_enu**2)
        if vel_mag > 0.1:
            self.get_logger().info(f'🎯 Pos: ({pos_x_enu:.2f}, {pos_y_enu:.2f}, {pos_z_enu:.2f}) | V: {vel_mag:.2f} m/s | Vel_z: {vel_z_enu:.2f}')

    def check_planner_timeout(self):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_planner_cmd_time).nanoseconds / 1e9

        if time_diff > self.planner_cmd_timeout:
            msg = TrajectorySetpoint()
            msg.velocity = [0.0, 0.0, 0.0]
            msg.position = [float('nan'), float('nan'), float('nan')]  # 호버링 시에도 z축 자유
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
            msg.yaw = float('nan')
            msg.yawspeed = float('nan')
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Offboard mode")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = True  # acceleration도 사용
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()
        self.offboard_setpoint_counter += 1

        if self.current_state == self.STATE_PLANNER_READY:
            self.check_planner_timeout()

        if self.current_state == self.STATE_INIT and self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.current_state = self.STATE_ARMING

        elif self.current_state == self.STATE_ARMING and self.offboard_setpoint_counter == 20:
            self.arm()
            self.current_state = self.STATE_TAKEOFF

        elif self.current_state == self.STATE_TAKEOFF:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

            if self.odometry_data_valid and abs(self.current_position[2] - (-self.takeoff_height)) < self.tolerance:
                self.current_state = self.STATE_PLANNER_READY
                self.get_logger().info(f"🚁 Ready! Pos: {self.current_position}")
                # self.get_logger().info(f"📊 PointCloud: {"✅" if self.pointcloud_received else "❌"}")

def main(args=None) -> None:
    print('Starting EGO-Planner + PX4 controller...')
    rclpy.init(args=args)
    drone_controller = EgoPlannerPX4Controller()
    rclpy.spin(drone_controller)
    drone_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
