"""
PPO Training Pipeline for Local Driving Navigation Agent.
Uses Stable-Baselines3 with the realistic 2-way metropolitan city road network.
Features:
- Dynamic combinatorial training across all 506 Origin-Destination pairs in the city.
- Evaluates on diverse randomized test routes to ensure generalizable driving skills.
- Multi-lane discipline, obstacle avoidance, overtaking, and traffic light compliance.
"""

import os
import random
from typing import Optional, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.local_env.environment import LocalDrivingEnv
from src.global_routing.realistic_city import RealisticCityManager
from src.global_routing.router import GlobalRouter


class RouteProgressCallback(BaseCallback):
    """Logs the OD trip pairs sampled during training episodes."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if an episode just finished
        dones = self.locals.get("dones", [False])
        if any(dones):
            self.episode_count += 1
            infos = self.locals.get("infos", [{}])
            for info in infos:
                if "stats" in info:
                    stats = info["stats"]
                    c_reason = stats.get("collision_reason", "")
                    spd = stats.get("max_speed_kmh", 0.0)
                    overtakes = stats.get("overtake_count", 0)
                    if self.verbose > 0 and self.episode_count % 10 == 0:
                        status_str = "SUCCESS" if info.get("reached_destination", False) else f"FAIL ({c_reason})"
                        print(f"[EPISODE #{self.episode_count}] Kết quả: {status_str} | V_max: {spd:.1f} km/h | Vượt xe: {overtakes}")
        return True


def make_city_driving_env(
    city_manager: RealisticCityManager,
    auto_sample_route: bool = True,
    traffic_density: str = "MEDIUM",
    log_dir: Optional[str] = None
) -> Monitor:
    """Creates a LocalDrivingEnv initialized with the full 23-node metropolitan network."""
    env = LocalDrivingEnv(
        network_manager=city_manager,
        auto_sample_route=auto_sample_route,
        num_npcs=8,
        radar_range=45.0,
        dt=0.1,
        max_steps=750
    )
    env.traffic_density_mode = traffic_density
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, log_dir)
    return Monitor(env)


def train_ppo_agent(
    total_timesteps: int = 200_000,
    save_dir: str = "models/saved_models",
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "logs/",
    model_name: str = "ppo_autonomous_driver"
) -> PPO:
    """
    Train PPO agent across combinatorial Origin-Destination routes throughout the entire city.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Khởi tạo bản đồ đô thị hoàn chỉnh (Realistic Metropolitan Network - 23 nút giao, 506 cặp hành trình)
    city_manager = RealisticCityManager.create_metropolitan_city()
    print(f"[MAP INITIALIZED] Khoi tao ban do do thi: {len(city_manager.graph.nodes())} nut giao, 2 chieu - 3 lan duong.")

    # 2. Khởi tạo môi trường huấn luyện với chế độ TỰ ĐỘNG CHỌN TỔ HỢP ĐIỂM ĐI - ĐIỂM ĐẾN
    print("[ENV SETUP] Kich hoat che do tu dong lay mau lo trinh to hop (506 cap Start -> Goal)...")
    train_env = make_city_driving_env(city_manager, auto_sample_route=True, traffic_density="MEDIUM", log_dir=log_dir)

    # 3. Khởi tạo môi trường đánh giá độc lập (Evaluation Env)
    eval_env = make_city_driving_env(city_manager, auto_sample_route=True, traffic_density="LOW")

    # 4. Thiết lập Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix=model_name
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False
    )

    progress_callback = RouteProgressCallback(verbose=1)

    # 5. Khởi tạo tác nhân PPO với cấu hình mạng nơ-ron học sâu
    print("Khoi tao PPO Agent voi MlpPolicy va mang Actor-Critic [128, 128]...")
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # Check if tensorboard is installed
    try:
        import tensorboard
        tb_log = log_dir
    except ImportError:
        tb_log = None

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log
    )

    # 6. Bắt đầu quá trình huấn luyện
    print(f"Bat dau huan luyen RL PPO cho {total_timesteps:,} buoc...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, progress_callback]
    )

    # 7. Lưu mô hình hoàn tất
    final_path = os.path.join(save_dir, f"{model_name}_final.zip")
    model.save(final_path)
    print(f"\n[HOAN TAT] Huan luyen thanh cong! Model da luu tai: {final_path}")
    print(f"[BEST MODEL] Model dat diem cao nhat luu tai: {save_dir}/best_model.zip")

    return model


if __name__ == "__main__":
    train_ppo_agent(total_timesteps=100_000)