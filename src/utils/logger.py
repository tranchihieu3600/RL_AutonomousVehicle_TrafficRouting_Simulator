"""
Run Logger module: Records telemetry, status, violations, and replay data for each trial run.
Logs are stored in `logs/runs/` as timestamped JSON files and appended to an index summary CSV.
"""

from typing import Dict, Any, Optional
import os
import json
import csv
from datetime import datetime
import numpy as np


class RunLogger:
    def __init__(self, log_dir: str = "logs/runs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_index_path = os.path.join(self.log_dir, "runs_summary.csv")
        self._init_csv()

    def _init_csv(self) -> None:
        if not os.path.exists(self.csv_index_path):
            with open(self.csv_index_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id",
                    "timestamp",
                    "status",
                    "retry_count",
                    "start_node",
                    "goal_node",
                    "route_algorithm",
                    "traffic_condition",
                    "max_speed_kmh",
                    "avg_speed_kmh",
                    "overtakes",
                    "red_light_violations",
                    "yellow_light_violations",
                    "collision_reason",
                    "log_file"
                ])

    def log_trial_run(
        self,
        status: str,  # "SUCCESS" or "CRASHED"
        start_node: str,
        goal_node: str,
        algorithm_name: str,
        retry_count: int,
        deterministic_traffic: bool,
        env_stats: Dict[str, Any],
        collision_reason: str = ""
    ) -> str:
        """
        Ghi log chi tiết cho mỗi lần chạy thử và lưu vào thư mục logs/runs/
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        run_id = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        log_filename = f"run_{run_id}_{status.lower()}.json"
        log_file_path = os.path.join(self.log_dir, log_filename)

        speeds = env_stats.get("speeds_recorded", [0.0])
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(env_stats.get("max_speed_kmh", 0.0))
        overtakes = int(env_stats.get("overtake_count", 0))
        red_viols = int(env_stats.get("red_light_violations", 0))
        yellow_viols = int(env_stats.get("yellow_light_violations", 0))
        step_count = int(env_stats.get("step_count", 0))

        run_data = {
            "run_id": run_id,
            "timestamp": timestamp_str,
            "status": status,
            "retry_count": retry_count,
            "start_node": start_node,
            "goal_node": goal_node,
            "route_algorithm": algorithm_name,
            "traffic_condition": "CỐ ĐỊNH (Deterministic)" if deterministic_traffic else "NGẪU NHIÊN (Random)",
            "telemetry": {
                "step_count": step_count,
                "max_speed_kmh": round(max_speed, 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "overtake_count": overtakes,
            },
            "traffic_light_violations": {
                "red_light_violations": red_viols,
                "yellow_light_violations": yellow_viols
            },
            "failure_reason": collision_reason if status == "CRASHED" else None
        }

        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4, ensure_ascii=False)

        with open(self.csv_index_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id,
                timestamp_str,
                status,
                retry_count,
                start_node,
                goal_node,
                algorithm_name,
                "Deterministic" if deterministic_traffic else "Random",
                f"{max_speed:.1f}",
                f"{avg_speed:.1f}",
                overtakes,
                red_viols,
                yellow_viols,
                collision_reason,
                log_filename
            ])

        print(f"📝 [LOG RUN] Đã lưu log chạy kiểm chứng tại: {log_file_path}")
        return log_file_path


# Global instance
run_logger = RunLogger()
