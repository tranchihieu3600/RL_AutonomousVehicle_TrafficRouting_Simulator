import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.global_routing.realistic_city import RealisticCityManager
from src.integration.hybrid_controller import HybridNavigationController
from src.visualization.pygame_renderer import PygameRenderer
from src.utils.logger import run_logger
from src.utils.evaluation_tracker import TrainingResultTracker


class OnlinePPOAgent:
    """
    Lightweight, high-speed PyTorch Actor-Critic Agent for Online Interactive Training.
    Allows continuous gradient backpropagation inside the Pygame simulation loop.
    """
    def __init__(self, state_dim: int = 17, action_dim: int = 2, lr: float = 3e-4, model_path: str = "models/saved_models/ppo_autonomous_driver.pt"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = model_path
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Actor Network (Outputs mean action for Steering & Throttle)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        # Critic Network (Value estimator)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buffer = []
        self.update_count = 0

        # Load weights if checkpoint exists
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path)
                self.actor.load_state_dict(ckpt["actor"])
                self.critic.load_state_dict(ckpt["critic"])
                self.update_count = ckpt.get("update_count", 0)
                print(f"🧠 [PPO LOADED] Đã nạp trọng số mô hình từ {self.model_path} (Lần cập nhật: {self.update_count})")
            except Exception as e:
                print(f"[PPO INIT] Khởi tạo mô hình mới: {e}")

    def select_action(self, state: np.ndarray, explore_noise: float = 0.1) -> np.ndarray:
        s_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean = self.actor(s_tensor).squeeze(0).numpy()
        noise = np.random.normal(0, explore_noise, size=self.action_dim)
        action = np.clip(action_mean + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update_weights(self, gamma: float = 0.99, epochs: int = 4) -> float:
        if len(self.buffer) < 16:
            return 0.0

        states = torch.FloatTensor(np.array([b[0] for b in self.buffer]))
        actions = torch.FloatTensor(np.array([b[1] for b in self.buffer]))
        rewards = [b[2] for b in self.buffer]
        dones = [b[4] for b in self.buffer]

        # Compute discounted returns
        discounted_rewards = []
        running_add = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                running_add = 0.0
            running_add = r + gamma * running_add
            discounted_rewards.insert(0, running_add)
        returns = torch.FloatTensor(discounted_rewards).unsqueeze(1)

        total_loss = 0.0
        for _ in range(epochs):
            values = self.critic(states)
            advantages = (returns - values).detach()

            action_pred = self.actor(states)
            actor_loss = -torch.mean(action_pred * actions * advantages)
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.update_count += 1
        self.buffer.clear()

        # Save weights every 5 updates
        if self.update_count % 5 == 0:
            torch.save({
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "update_count": self.update_count
            }, self.model_path)
            print(f"💾 [MODEL CHECKPOINT] Đã lưu trọng số PPO cập nhật #{self.update_count} vào {self.model_path}")

        return total_loss / epochs


def run_hybrid_simulation():
    print("=========================================================================")
    print(" SMART AUTONOMOUS DRIVING: TURN SIGNALS & INTERACTIVE JOURNEY DASHBOARD")
    print("=========================================================================")

    city = RealisticCityManager.create_metropolitan_city()

    start_node = "Port_Harbor"
    goal_node = "Airport_Terminal"

    coordinator = HybridNavigationController(city, start_node=start_node, goal_node=goal_node)
    env = coordinator.env
    obs, _ = env.reset()

    # Track evaluation analytics with 100 episodes per data point and export to `results/`
    tracker = TrainingResultTracker(batch_size=100, output_dir="results")

    renderer = PygameRenderer(coordinator, width=1400, height=800)
    renderer.on_close_callback = lambda: tracker.generate_and_save_report()
    renderer.on_save_report_callback = lambda: tracker.generate_and_save_report()

    # Online Reinforcement Learning Agent
    rl_agent = OnlinePPOAgent(state_dim=17, action_dim=2)

    print("\n[CHẾ ĐỘ CHỜ CẤU HÌNH LỘ TRÌNH]:")
    print("1. Bấm nút màu xanh [▶ BẮT ĐẦU CHẠY] để xe bắt đầu hành trình.")
    print("2. Bấm nút [🧠 RL HỌC ONLINE: BẬT] để bật mạng nơ-ron tự học và tự cập nhật trọng số.")
    print("3. Chọn các tốc độ 100x, 1000x, 5000x để huấn luyện siêu tốc!")

    while True:
        # Only advance simulation physics if user pressed 'BẮT ĐẦU CHẠY' (sim_state == 'RUNNING')
        if renderer.sim_state == "RUNNING":
            multiplier = getattr(renderer, "sim_speed_multiplier", 1.0)
            sub_steps = int(multiplier) if multiplier >= 2.0 else 1

            for _ in range(sub_steps):
                if renderer.sim_state != "RUNNING":
                    break
                coordinator.update_cycle()

                # Kiểm tra chế độ điều khiển:
                # Khi [RL HỌC ONLINE] BẬT: Dùng mô hình RL lai (Residual RL / Guided Action)
                # Giúp mạng nơ-ron có đà xuất phát (baseline cruising), học tăng tốc/né vật cản mà KHÔNG bị đứng im!
                if renderer.rl_training_enabled:
                    raw_action = rl_agent.select_action(obs, explore_noise=0.08)
                    # Ga cơ sở: nếu đường phía trước hoàn toàn thông thoáng, đảm bảo ga dương tối thiểu +0.25
                    front_dist_chk, _, _ = env.check_traffic_ahead_and_overtake_lane()
                    tl_dist_chk, tl_state_chk, _ = env.check_upcoming_traffic_light()
                    is_path_clear = (front_dist_chk > 20.0) and not (tl_state_chk in ["RED", "YELLOW"] and tl_dist_chk < 25.0)

                    ai_throttle = float(raw_action[1])
                    ego_spd = env.ego_vehicle.speed
                    # Chỉ hỗ trợ mớm ga ban đầu khi xe đang đi chậm (< 12 m/s ~ 43 km/h), đường cực kỳ thông thoáng (> 25m)
                    if is_path_clear and ego_spd < 12.0 and front_dist_chk > 25.0 and ai_throttle < 0.15:
                        blended_throttle = max(0.2, 0.5 * ai_throttle + 0.3)
                    elif ego_spd > 15.0:  # Đã tiệm cận 54 km/h (gần giới hạn 60 km/h) -> cấm bù ga dương
                        blended_throttle = min(0.0, ai_throttle)
                    else:
                        blended_throttle = ai_throttle

                    action = np.array([raw_action[0], blended_throttle], dtype=np.float32)
                else:

                    # Góc lái được tự động điều phối chuẩn xác và chống xoay vòng trong env.step()
                    steer_action = 0.0

                    front_dist, passing_clear, front_npc = env.check_traffic_ahead_and_overtake_lane()
                    tl_dist, tl_state, tl_timer = env.check_upcoming_traffic_light()

                    if front_dist < 16.0:
                        front_spd = front_npc.speed if front_npc else 0.0
                        if front_dist < 6.5:
                            throttle_action = -1.0 if ego.speed > 0.2 else -0.1
                            if front_dist < 4.0 and ego.speed > 0.1:
                                ego.speed = max(0.0, ego.speed - 1.2)
                        elif front_dist < 10.0:
                            target_v = max(0.0, front_spd - 1.0)
                            throttle_action = -0.7 if ego.speed > target_v else 0.0
                        else:
                            target_v = max(0.0, front_spd)
                            throttle_action = -0.4 if ego.speed > target_v else 0.1

                    elif tl_state in ["RED", "YELLOW"] and 1.5 < tl_dist < 26.0:
                        brake_intensity = -0.85 if tl_dist < 10.0 else -0.45
                        throttle_action = brake_intensity if ego.speed > 0.3 else -0.1

                    elif ego.overtaking_state in ["CHANGING_OUTER", "PASSING", "CHANGING_INNER"]:
                        throttle_action = 0.5 if ego.speed < 15.0 else -0.1
                    else:
                        throttle_action = 0.35 if ego.speed < 11.0 else -0.1

                    action = np.array([steer_action, throttle_action], dtype=np.float32)

                next_obs, reward, terminated, truncated, info = env.step(action)

                # Thu thập dữ liệu và cập nhật trọng số Online nếu bật chế độ RL
                if renderer.rl_training_enabled:
                    rl_agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
                    renderer.rl_total_timesteps += 1
                    # Cập nhật lan truyền ngược gradient định kỳ mỗi 64 bước hoặc khi tập kết thúc
                    if len(rl_agent.buffer) >= 64 or terminated or truncated:
                        rl_agent.update_weights(epochs=3)
                        renderer.rl_updates_count = rl_agent.update_count

                obs = next_obs


                if terminated:
                    algo_name = coordinator.route_proposals[coordinator.active_proposal_index].get("name", "Unknown") if coordinator.route_proposals else "A*"
                    renderer.training_episodes_completed += 1
                    is_success = bool(info.get("reached_destination", False))

                    # Feed episode statistics into 100-batch evaluator tracker
                    spd_list = env.stats.get("speeds_recorded", [0.0])
                    mean_spd = float(np.mean(spd_list)) if spd_list else 0.0
                    tracker.record_episode(
                        episode_idx=renderer.training_episodes_completed,
                        is_success=is_success,
                        steps=int(env.stats.get("step_count", 0)),
                        avg_speed_kmh=mean_spd,
                        overtakes=int(env.stats.get("overtake_count", 0)),
                        red_violations=int(env.stats.get("red_light_violations", 0)),
                        yellow_violations=int(env.stats.get("yellow_light_violations", 0)),
                        origin=coordinator.start_node,
                        destination=coordinator.goal_node,
                        algorithm=algo_name
                    )

                    if is_success:
                        renderer.training_success_count += 1
                        print(f"\n[CHÚC MỪNG] Đã đến đích an toàn tại {coordinator.goal_node}!")
                        run_logger.log_trial_run(
                            status="SUCCESS",
                            start_node=coordinator.start_node,
                            goal_node=coordinator.goal_node,
                            algorithm_name=algo_name,
                            retry_count=renderer.retry_count,
                            deterministic_traffic=renderer.deterministic_traffic,
                            env_stats=env.stats
                        )

                        # Khi bật [AUTO-RUN] hoặc [RL HỌC ONLINE]: Tự động chuyển ngay sang lộ trình tổ hợp mới
                        is_training_active = renderer.auto_training_mode or renderer.rl_training_enabled
                        if is_training_active:
                            active_seed = renderer.traffic_seed if renderer.deterministic_traffic else None
                            new_s, new_g = coordinator.setup_random_training_route(deterministic_seed=active_seed)
                            renderer.retry_count = 0
                            renderer.sim_state = "RUNNING"
                            if hasattr(env, "ego_vehicle"):
                                renderer.cam_x = env.ego_vehicle.x
                                renderer.cam_y = env.ego_vehicle.y
                            obs, _ = env.reset(seed=active_seed)
                            print(f"⚡ [RL TO HOP LO TRINH] Hoan thanh! Chuyen sang tuyen duong moi: {new_s} -> {new_g}")
                        else:
                            renderer.sim_state = "FINISHED"

                    elif info.get("is_crashed", False) or info.get("is_stagnated", False) or info.get("is_red_light_fail", False) or info.get("is_off_road", False):
                        renderer.retry_count += 1
                        renderer.last_crash_reason = info.get("collision_reason", "Thất bại lộ trình!")
                        print(f"\n[THẤT BẠI #{renderer.retry_count}] {renderer.last_crash_reason}")

                        run_logger.log_trial_run(
                            status="FAILED",
                            start_node=coordinator.start_node,
                            goal_node=coordinator.goal_node,
                            algorithm_name=algo_name,
                            retry_count=renderer.retry_count,
                            deterministic_traffic=renderer.deterministic_traffic,
                            env_stats=env.stats,
                            collision_reason=renderer.last_crash_reason
                        )

                        is_training_active = renderer.auto_training_mode or renderer.rl_training_enabled
                        if is_training_active or renderer.auto_retry_until_success:
                            active_seed = renderer.traffic_seed if renderer.deterministic_traffic else None
                            # Nếu trong chế độ RL hoặc Auto-run mà thử lại quá 2 lần không thành công,
                            # chuyển sang một lộ trình tổ hợp ngẫu nhiên mới để AI không bị mắc kẹt tại một điểm
                            if is_training_active and renderer.retry_count >= 2 and not renderer.auto_retry_until_success:
                                new_s, new_g = coordinator.setup_random_training_route(deterministic_seed=active_seed)
                                renderer.retry_count = 0
                                obs, _ = env.reset(seed=active_seed)
                                print(f"🔁 [RL TO HOP LO TRINH] Chuyen sang tuyen duong to hop moi: {new_s} -> {new_g}")
                            else:
                                obs, _ = env.reset(seed=active_seed)
                                mode_lbl = "RL-TRAINING" if renderer.rl_training_enabled else ("AUTO-TRAIN" if renderer.auto_training_mode else "AUTO-RETRY")
                                cond_lbl = "Cố định" if renderer.deterministic_traffic else "Ngẫu nhiên"
                                print(f"🔁 [{mode_lbl}] Tự động thử lại lần #{renderer.retry_count} (Điều kiện: {cond_lbl})...")
                            renderer.sim_state = "RUNNING"
                        else:
                            renderer.sim_state = "CRASHED"

                    break

        # Always render frame and process UI events
        renderer.render_frame()


if __name__ == "__main__":
    run_hybrid_simulation()
