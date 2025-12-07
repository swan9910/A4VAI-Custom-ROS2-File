
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # 드론 ID
    drone_id = LaunchConfiguration('drone_id', default='0')
    # 맵 크기 - ⭐ 크게 증가! ⭐
    map_size_x = LaunchConfiguration('map_size_x', default='220.0')  # 50 → 200
    map_size_y = LaunchConfiguration('map_size_y', default='220.0')  # 50 → 200
    map_size_z = LaunchConfiguration('map_size_z', default='50.0')   # 10 → 20

    # 드론 성능 파라미터 (S2000용)
    max_vel = LaunchConfiguration('max_vel', default='2.5')
    max_acc = LaunchConfiguration('max_acc', default='2.0')
    # EGO-Planner 노드
    ego_planner_node = Node(
        package='ego_planner',
        executable='ego_planner_node',
        name='ego_planner_node',
        output='screen',
        remappings=[
            ('odom_world', '/ego_odom'),
            ('grid_map/odom', '/ego_odom_grid'),
            ('grid_map/pose', '/ego_pose'),
            ('grid_map/cloud', '/camera/depth/points'),
            ('planning/bspline', '/planning/bspline'),
            ('planning/data_display', '/planning/data_display'),
            ('planning/broadcast_bspline_from_planner', '/broadcast_bspline'),
            ('planning/broadcast_bspline_to_planner', '/broadcast_bspline'),
        ],
        parameters=[
            # FSM 설정
            {'fsm/flight_type': 1},
            {'fsm/thresh_replan_time': 1.0},
            {'fsm/thresh_no_replan_meter': 2.0},
            {'fsm/planning_horizon': 40.0},
            {'fsm/planning_horizen_time': 3.0},
            {'fsm/emergency_time': 1.0},
            {'fsm/realworld_experiment': False},
            {'fsm/fail_safe': True},

            # Grid Map 설정 - ⭐ 큰 맵용 ⭐
            {'grid_map/resolution': 1.0},  # 0.15 → 0.2 (큰 맵이라 해상도 약간 감소)
            {'grid_map/map_size_x': map_size_x},
            {'grid_map/map_size_y': map_size_y},
            {'grid_map/map_size_z': map_size_z},
            {'grid_map/local_update_range_x': 50.0},  # 8 → 12 (넓은 범위)
            {'grid_map/local_update_range_y': 50.0},
            {'grid_map/local_update_range_z': 30.0},   # 5 → 6
            {'grid_map/obstacles_inflation': 3.0},
            {'grid_map/local_map_margin': 20},  # 15 → 20
            {'grid_map/ground_height': 1.5},

            # 카메라 파라미터
            {'grid_map/cx': 320.0},
            {'grid_map/cy': 240.0},
            {'grid_map/fx': 320.0},
            {'grid_map/fy': 320.0},

            # Depth filter - 드론 자체를 감지하지 않도록
            {'grid_map/use_depth_filter': False},
            {'grid_map/depth_filter_tolerance': 0.2},
            {'grid_map/depth_filter_maxdist': 40.0},  # 12 → 15 (더 멀리)
            {'grid_map/depth_filter_mindist': 1.0},   # 1.2 → 1.5 (더 안전하게)
            {'grid_map/depth_filter_margin': 2},
            {'grid_map/k_depth_scaling_factor': 1000.0},
            {'grid_map/skip_pixel': 2},

            # Local fusion
            {'grid_map/p_hit': 0.65},
            {'grid_map/p_miss': 0.30},
            {'grid_map/p_min': 0.12},
            {'grid_map/p_max': 0.90},
            {'grid_map/p_occ': 0.80},
            {'grid_map/min_ray_length': 1.0},  # 1.2 → 1.5
            {'grid_map/max_ray_length': 15.0},  # 12 → 15

            {'grid_map/virtual_ceil_height': 10.0},
            {'grid_map/visualization_truncate_height': 10.0},
            {'grid_map/show_occ_time': False},
            {'grid_map/pose_type': 2},
            {'grid_map/frame_id': 'world'},

            # Planner Manager (S2000용)
            {'manager/max_vel': max_vel},
            {'manager/max_acc': max_acc},
            {'manager/max_jerk': 3.0},
            {'manager/control_points_distance': 0.5},
            {'manager/feasibility_tolerance': 0.08},
            {'manager/planning_horizon': 40.0},  # 10 → 12 (더 멀리 계획)
            {'manager/use_distinctive_trajs': False},
            {'manager/drone_id': 0},

            {'optimization/lambda_smooth': 2.0},
            {'optimization/lambda_collision': 2.5},
            {'optimization/lambda_feasibility': 0.1},
            {'optimization/lambda_fitness': 0.1},
            {'optimization/dist0': 1.5},
            {'optimization/swarm_clearance': 0.5},
            {'optimization/max_vel': max_vel},
            {'optimization/max_acc': max_acc},

            # B-Spline
            {'bspline/limit_vel': max_vel},
            {'bspline/limit_acc': max_acc},
            {'bspline/limit_ratio': 1.1},
        ]
    )

    # Trajectory Server 노드
    traj_server_node = Node(
        package='ego_planner',
        executable='traj_server',
        name='traj_server',
        output='screen',
        remappings=[
            ('position_cmd', '/planning/pos_cmd'),
            ('planning/bspline', '/planning/bspline')
        ],
        parameters=[
            {'traj_server/time_forward': 1.0}
        ]
    )

    # LaunchDescription 생성
    ld = LaunchDescription()

    # Declare arguments
    ld.add_action(DeclareLaunchArgument('drone_id', default_value='0'))
    ld.add_action(DeclareLaunchArgument('map_size_x', default_value='220.0'))
    ld.add_action(DeclareLaunchArgument('map_size_y', default_value='220.0'))
    ld.add_action(DeclareLaunchArgument('map_size_z', default_value='30.0'))
    ld.add_action(DeclareLaunchArgument('max_vel', default_value='4.0'))
    ld.add_action(DeclareLaunchArgument('max_acc', default_value='1.0'))

    # Add nodes
    ld.add_action(ego_planner_node)
    ld.add_action(traj_server_node)

    return ld
