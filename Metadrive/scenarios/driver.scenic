param map = localPath('../CARLA/Town01.xodr')
param carla_map = 'Town01'

model scenic.simulators.metadrive.model

param time_step = 1.0 / 10
param verifaiSamplerType = 'ce'
param render = 0
param use2DMap = True

# traffic 稍微多一点
param extra_cars = 4

import numpy as np
TERMINATE_TIME = 40 / globalParameters.time_step

"""
Global params for road / lane / starting positions.
"""

def get_nearest_centerline(obj):
    min_dist = np.inf
    for lane in network.lanes:
        dist = distance to lane
        if dist < min_dist:
            min_dist = dist
            centerline = lane.centerline
    return centerline

# 随机选路 / 车道
param select_road = Uniform(*network.roads)
param distractor_road = Uniform(*network.roads)

param select_lane = Uniform(*network.lanes)
param distractor_lane = Uniform(*network.lanes)

# ego 出生点
start = Uniform(*globalParameters.select_lane.centerline.points)
start = (start[0] @ start[1])

# distractor 出生点
start2 = Uniform(*globalParameters.distractor_lane.centerline.points)
start2 = (start2[0] @ start2[1])

# ego：相对 roadDirection 加一个随机偏转（±0.25 rad）
ego = new Car on start,
    facing (Uniform(-0.25, 0.25) relative to roadDirection),
    with observation 0, with cte 0

# lead 车：在 ego 前方 8~20 米
leadCar = new Car ahead of ego by Uniform(8, 20),
    with behavior DriveAvoidingCollisions(
        target_speed = Uniform(15, 25),
        avoidance_threshold = Uniform(4, 10)
    )

# 额外车流
for id in range(globalParameters.extra_cars):
    new Car with behavior DriveAvoidingCollisions()

# 背景干扰车：在另一条随机车道上跑
distractor = new Car on start2,
    with behavior DriveAvoidingCollisions(
        target_speed = 15,
        avoidance_threshold = 12
    )

monitor DrivingReward(obj):
    while True:
        ego.previous_coordinates = obj.position
        lane = obj._lane

        if lane:
            centerline = lane.centerline
        else:
            centerline = get_nearest_centerline(obj)

        if obj._lane:
            ego.lane_heading = lane._defaultHeadingAt(ego.position)
            orientation_error = np.abs(
                ego.heading - lane._defaultHeadingAt(ego.position)
            )
            ego.orientation_error = orientation_error

            if orientation_error > .05:
                orientation_error = max(-orientation_error, -3)
            else:
                orientation_error = 0

        nearest_line_points = centerline.nearestSegmentTo(obj.position)
        nearest_line_segment = PolylineRegion(nearest_line_points)

        cte = min(abs(distance to nearest_line_segment), 1)

        # 更敏感，让 coverage 数值拉开
        if cte < 0.05:
            cte = 0

        ego.cte = cte

        speed_reward = max(0.05 * ego.speed, 1/2)
        dist_reward = distance to ego.previous_coordinates

        reward = -cte + speed_reward + orientation_error + dist_reward
        ego.reward = reward

        wait

require monitor DrivingReward(ego)
