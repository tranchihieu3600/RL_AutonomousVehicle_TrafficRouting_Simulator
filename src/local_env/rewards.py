"""
Reward and penalty calculation for Local Reinforcement Learning agent.
"""

from typing import Dict, Any
import math


class RewardCalculator:
    """Computes dense reward signal to encourage safe driving, waypoint following, and comfort."""

    def __init__(
        self,
        w_progress: float = 1.5,
        w_speed: float = 0.5,
        w_lane_align: float = 0.3,
        penalty_collision: float = 500.0,
        penalty_jerk: float = 0.05,
        penalty_red_light: float = 500.0,
        penalty_yellow_light: float = 150.0,
        target_speed: float = 12.0  # ~43 km/h
    ):
        self.w_progress = w_progress
        self.w_speed = w_speed
        self.w_lane_align = w_lane_align
        self.penalty_collision = penalty_collision
        self.penalty_jerk = penalty_jerk
        self.penalty_red_light = penalty_red_light
        self.penalty_yellow_light = penalty_yellow_light
        self.target_speed = target_speed

    def compute_reward(
        self,
        prev_dist_to_wp: float,
        curr_dist_to_wp: float,
        current_speed: float,
        heading_error: float,
        is_crashed: bool,
        is_reached_wp: bool,
        delta_action: float = 0.0,
        traffic_light_violation: str = "NONE",  # "NONE", "RED", "YELLOW"
        front_gap: float = 999.0,
        rel_speed: float = 0.0,  # ego_speed - front_speed
        tl_dist: float = 999.0,
        tl_state: str = "NONE",
        stagnation_steps: int = 0
    ) -> float:
        """Calculate total step reward with TTC penalty, stop bonus, and severe violation penalties."""
        if is_crashed:
            return -self.penalty_collision

        # Severe penalty on red light violation
        if traffic_light_violation == "RED":
            return -self.penalty_red_light
        elif traffic_light_violation == "YELLOW":
            return -self.penalty_yellow_light

        # Progress reward: positive when moving closer to waypoint
        r_progress = (prev_dist_to_wp - curr_dist_to_wp) * self.w_progress

        # Speed reward: reward driving near target speed, heavily penalize speeding over 60 km/h
        # Tốc độ lý tưởng: 40 - 55 km/h (~11 - 15 m/s); Quá 60 km/h (16.67 m/s) bị phạt nặng!
        speed_kmh = current_speed * 3.6
        r_speed = 0.0
        r_speeding_penalty = 0.0

        if speed_kmh > 62.0:
            # Phạt vượt quá tốc độ biển báo (quá 60-62 km/h)
            overshoot = speed_kmh - 60.0
            r_speeding_penalty = -1.5 * overshoot
        else:
            speed_ratio = current_speed / max(1e-3, self.target_speed)
            r_speed = self.w_speed * (1.0 - abs(1.0 - min(speed_ratio, 1.3)))

        # Heading alignment reward: reward looking directly at target waypoint
        r_align = self.w_lane_align * math.cos(heading_error)

        # Action smoothness penalty
        r_jerk = -self.penalty_jerk * abs(delta_action)


        # Bonus when reaching a waypoint
        r_wp_bonus = 20.0 if is_reached_wp else 0.0

        # Method 3: Time-To-Collision (TTC) exponential penalty
        r_ttc = 0.0
        if rel_speed > 0.1 and front_gap < 30.0:
            ttc = front_gap / max(0.1, rel_speed)
            if ttc < 2.5:
                # Exponential penalty: the closer TTC is to 0, the more devastating the penalty
                r_ttc = -2.5 * math.exp(2.5 - ttc)

        # Method 3: Stop bonus vs Stagnation Penalty
        # Được thưởng dừng đỗ kiên nhẫn nếu THỰC SỰ có xe chắn phía trước hoặc gặp đèn đỏ/đèn vàng
        is_stopped = current_speed < 0.5
        waiting_behind_traffic = (front_gap < 22.0 and is_stopped)
        waiting_at_red = (tl_state in ["RED", "YELLOW"] and tl_dist < 32.0 and is_stopped)

        r_stop_bonus = 0.0
        r_stagnation_penalty = 0.0

        if waiting_behind_traffic or waiting_at_red:
            r_stop_bonus = 0.5  # Thưởng đỗ kiên nhẫn khi gặp chướng ngại vật/đèn đỏ
        elif is_stopped and stagnation_steps >= 50:
            # Chỉ phạt nặng nếu đứng yên quá 5 giây khi đường hoàn toàn thông thoáng!
            r_stagnation_penalty = -3.0

        total_reward = (
            r_progress
            + r_speed
            + r_speeding_penalty
            + r_align
            + r_jerk
            + r_wp_bonus
            + r_ttc
            + r_stop_bonus
            + r_stagnation_penalty
        )
        return float(total_reward)



