"""
Evaluation & Analytics Evaluator:
Aggregates training episodes in batches (e.g., every 100 episodes as 1 data point).
Computes:
- Success rate (Độ tin cậy hoàn thành hành trình)
- Wilson Score 95% Confidence Interval (Khoảng tin cậy thống kê)
- Average speed & Standard deviation
- Collision rate and Traffic light compliance rate
Generates comprehensive visual plots and summary reports saved in `results/`.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import math
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Headless backend for thread-safe rendering
import matplotlib.pyplot as plt


def calculate_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson Score Interval for binomial proportion confidence.
    Returns (center_estimate, lower_bound, upper_bound).
    """
    if total <= 0:
        return 0.0, 0.0, 0.0

    p = float(successes) / float(total)
    # z = 1.96 for 95% confidence
    z = 1.96 if abs(confidence - 0.95) < 0.01 else 1.645
    z2 = z * z
    n = float(total)

    denominator = 1.0 + z2 / n
    centre_adjusted_probability = (p + z2 / (2.0 * n)) / denominator
    adjusted_std_error = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denominator

    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return p, lower, upper


class TrainingResultTracker:
    def __init__(self, batch_size: int = 100, output_dir: str = "results"):
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.episode_records: List[Dict[str, Any]] = []
        self.batch_points: List[Dict[str, Any]] = []

    def record_episode(
        self,
        episode_idx: int,
        is_success: bool,
        steps: int,
        avg_speed_kmh: float,
        overtakes: int,
        red_violations: int,
        yellow_violations: int,
        origin: str,
        destination: str,
        algorithm: str
    ) -> Optional[Dict[str, Any]]:
        """
        Record each episode. When reaching a multiple of batch_size (100, 200, 300...),
        computes an aggregated batch data point with confidence metrics.
        """
        record = {
            "episode": episode_idx,
            "success": is_success,
            "steps": steps,
            "avg_speed": avg_speed_kmh,
            "overtakes": overtakes,
            "red_violations": red_violations,
            "yellow_violations": yellow_violations,
            "origin": origin,
            "destination": destination,
            "algorithm": algorithm
        }
        self.episode_records.append(record)

        # Check if a 100-episode batch has completed
        if len(self.episode_records) % self.batch_size == 0:
            batch_point = self._compute_batch_point(len(self.episode_records) // self.batch_size)
            self.batch_points.append(batch_point)
            print(f"\n[TRAINING ANALYTICS] Da tich luy {len(self.episode_records)} lan chay ({batch_point['batch_name']}):")
            print(f"   * Ti le thanh cong (Do tin cay): {batch_point['success_rate']*100:.1f}% "
                  f"[95% CI: {batch_point['ci_lower']*100:.1f}% - {batch_point['ci_upper']*100:.1f}%]")
            print(f"   * Toc do trung binh: {batch_point['mean_speed']:.1f} +/- {batch_point['std_speed']:.1f} km/h")
            print(f"   * Vi pham den: {batch_point['total_violations']} lan | Vuot xe: {batch_point['total_overtakes']} lan")
            return batch_point
        return None

    def _compute_batch_point(self, batch_num: int) -> Dict[str, Any]:
        """Aggregates the most recent `self.batch_size` episodes."""
        start_idx = (batch_num - 1) * self.batch_size
        end_idx = batch_num * self.batch_size
        batch_slice = self.episode_records[start_idx:end_idx]

        total = len(batch_slice)
        successes = sum(1 for r in batch_slice if r["success"])
        p, lower, upper = calculate_wilson_score_interval(successes, total, confidence=0.95)

        speeds = [r["avg_speed"] for r in batch_slice if r["avg_speed"] > 0]
        mean_spd = float(np.mean(speeds)) if speeds else 0.0
        std_spd = float(np.std(speeds)) if speeds else 0.0

        total_ov = sum(r["overtakes"] for r in batch_slice)
        total_viols = sum(r["red_violations"] + r["yellow_violations"] for r in batch_slice)
        avg_steps = float(np.mean([r["steps"] for r in batch_slice]))

        return {
            "batch_index": batch_num,
            "batch_name": f"Batch {batch_num} ({start_idx+1}-{end_idx})",
            "episodes_count": total,
            "cumulative_episodes": end_idx,
            "successes": successes,
            "success_rate": p,
            "ci_lower": lower,
            "ci_upper": upper,
            "margin_error": (upper - lower) / 2.0,
            "mean_speed": mean_spd,
            "std_speed": std_spd,
            "total_overtakes": total_ov,
            "total_violations": total_viols,
            "mean_steps": avg_steps
        }

    def generate_and_save_report(self) -> Dict[str, str]:
        """
        Creates comprehensive graphs, summary metrics JSON, and a detailed Markdown report
        in `results/`.
        """
        if not self.episode_records:
            return {}

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # 1. Ensure at least one aggregated point exists if < 100 episodes
        if not self.batch_points:
            p, lower, upper = calculate_wilson_score_interval(
                sum(1 for r in self.episode_records if r["success"]),
                len(self.episode_records)
            )
            speeds = [r["avg_speed"] for r in self.episode_records if r["avg_speed"] > 0]
            self.batch_points.append({
                "batch_index": 1,
                "batch_name": f"Session (1-{len(self.episode_records)})",
                "episodes_count": len(self.episode_records),
                "cumulative_episodes": len(self.episode_records),
                "successes": sum(1 for r in self.episode_records if r["success"]),
                "success_rate": p,
                "ci_lower": lower,
                "ci_upper": upper,
                "margin_error": (upper - lower) / 2.0,
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "std_speed": float(np.std(speeds)) if speeds else 0.0,
                "total_overtakes": sum(r["overtakes"] for r in self.episode_records),
                "total_violations": sum(r["red_violations"] + r["yellow_violations"] for r in self.episode_records),
                "mean_steps": float(np.mean([r["steps"] for r in self.episode_records]))
            })

        # 2. Draw Multi-Metric Graphs (Figure with 4 subplots)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"BÁO CÁO KẾT QUẢ HUẤN LUYỆN TỰ HÀNH & ĐIỀU HƯỚNG GIAO THÔNG\n(Mỗi điểm đại diện cho {self.batch_size} lần thử nghiệm)", fontsize=15, fontweight="bold")

        x_vals = [bp["cumulative_episodes"] for bp in self.batch_points]
        x_labels = [bp["batch_name"] for bp in self.batch_points]

        # Plot 1: Success Rate & Wilson 95% Confidence Interval
        rates = [bp["success_rate"] * 100 for bp in self.batch_points]
        lowers = [bp["ci_lower"] * 100 for bp in self.batch_points]
        uppers = [bp["ci_upper"] * 100 for bp in self.batch_points]
        err_lower = [rates[i] - lowers[i] for i in range(len(rates))]
        err_upper = [uppers[i] - rates[i] for i in range(len(rates))]

        ax1 = axs[0, 0]
        ax1.errorbar(x_vals, rates, yerr=[err_lower, err_upper], fmt="-o", color="#00C853", ecolor="#69F0AE", elinewidth=2.5, capsize=6, capthick=2, label="Độ tin cậy (Tỉ lệ thành công)")
        ax1.fill_between(x_vals, lowers, uppers, color="#00C853", alpha=0.15, label="Khoảng tin cậy 95% (CI)")
        ax1.set_title("1. Độ tin cậy hoàn thành lộ trình (95% Confidence Interval)", fontweight="bold")
        ax1.set_xlabel("Số lần thử nghiệm tích lũy")
        ax1.set_ylabel("Tỉ lệ thành công (%)")
        ax1.set_ylim(-5, 105)
        ax1.legend(loc="lower right")

        # Plot 2: Average Speed & Speed Consistency
        mean_speeds = [bp["mean_speed"] for bp in self.batch_points]
        std_speeds = [bp["std_speed"] for bp in self.batch_points]
        ax2 = axs[0, 1]
        ax2.plot(x_vals, mean_speeds, "-s", color="#00B0FF", linewidth=2.5, label="Tốc độ trung bình (km/h)")
        ax2.fill_between(x_vals, [mean_speeds[i] - std_speeds[i] for i in range(len(mean_speeds))], [mean_speeds[i] + std_speeds[i] for i in range(len(mean_speeds))], color="#00B0FF", alpha=0.2, label="Độ lệch chuẩn tốc độ")
        ax2.set_title("2. Hiệu suất tốc độ hành trình", fontweight="bold")
        ax2.set_xlabel("Số lần thử nghiệm tích lũy")
        ax2.set_ylabel("Vận tốc (km/h)")
        ax2.legend(loc="upper left")

        # Plot 3: Overtaking Count & Traffic Light Violations
        overtakes = [bp["total_overtakes"] for bp in self.batch_points]
        violations = [bp["total_violations"] for bp in self.batch_points]
        ax3 = axs[1, 0]
        w = max(5, int(self.batch_size * 0.35))
        ax3.bar([x - w/2 for x in x_vals], overtakes, width=w, color="#FFD600", label="Số lần vượt xe thành công")
        ax3.bar([x + w/2 for x in x_vals], violations, width=w, color="#FF1744", label="Vi phạm đèn đỏ/vàng")
        ax3.set_title("3. Hành vi vượt xe & Tuân thủ luật giao thông", fontweight="bold")
        ax3.set_xlabel("Số lần thử nghiệm tích lũy")
        ax3.set_ylabel("Số lần phát sinh")
        ax3.legend(loc="upper right")

        # Plot 4: Average Navigation Steps to Destination
        mean_steps = [bp["mean_steps"] for bp in self.batch_points]
        ax4 = axs[1, 1]
        ax4.plot(x_vals, mean_steps, "-d", color="#AA00FF", linewidth=2.5, label="Số bước trung bình về đích")
        ax4.set_title("4. Tối ưu thời gian lộ trình (Số bước di chuyển)", fontweight="bold")
        ax4.set_xlabel("Số lần thử nghiệm tích lũy")
        ax4.set_ylabel("Steps trung bình")
        ax4.legend(loc="upper right")

        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, f"training_analytics_{timestamp}.png")
        latest_chart_path = os.path.join(self.output_dir, "training_analytics_latest.png")
        plt.savefig(chart_path, dpi=200)
        plt.savefig(latest_chart_path, dpi=200)
        plt.close(fig)

        # 3. Export Summary Metrics JSON
        json_path = os.path.join(self.output_dir, f"training_metrics_{timestamp}.json")
        latest_json_path = os.path.join(self.output_dir, "training_metrics_latest.json")
        overall_total = len(self.episode_records)
        overall_success = sum(1 for r in self.episode_records if r["success"])
        ov_p, ov_l, ov_u = calculate_wilson_score_interval(overall_success, overall_total)

        summary_data = {
            "timestamp": timestamp,
            "batch_size_per_point": self.batch_size,
            "total_episodes_evaluated": overall_total,
            "overall_success_count": overall_success,
            "overall_reliability": {
                "success_rate": round(ov_p, 4),
                "confidence_95_lower": round(ov_l, 4),
                "confidence_95_upper": round(ov_u, 4),
                "margin_of_error": round((ov_u - ov_l) / 2.0, 4)
            },
            "batch_data_points": self.batch_points
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)

        # 4. Export Markdown Summary
        md_path = os.path.join(self.output_dir, f"training_report_{timestamp}.md")
        latest_md_path = os.path.join(self.output_dir, "training_report_latest.md")
        md_content = self._generate_markdown_report(summary_data, chart_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(latest_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\n[RESULTS SAVED] Toan bo do thi va chi so danh gia da luu vao thu muc '{self.output_dir}':")
        print(f"   * Bieu do: {chart_path}")
        print(f"   * Du lieu JSON: {json_path}")
        print(f"   * Bao cao Markdown: {md_path}")

        return {
            "chart": chart_path,
            "json": json_path,
            "markdown": md_path
        }

    def _generate_markdown_report(self, data: Dict[str, Any], chart_path: str) -> str:
        ov = data["overall_reliability"]
        lines = [
            "# BÁO CÁO ĐÁNH GIÁ KẾT QUẢ HUẤN LUYỆN TỰ HÀNH",
            f"- **Thời gian khởi tạo:** {data['timestamp']}",
            f"- **Tổng số lần thử nghiệm:** {data['total_episodes_evaluated']}",
            f"- **Quy mô gom mẫu:** 1 điểm dữ liệu sau mỗi {data['batch_size_per_point']} lần thử nghiệm",
            "",
            "## 1. Chỉ Số Độ Tin Cậy & Xác Suất Thành Công",
            f"- **Tỉ lệ thành công tổng thể:** {ov['success_rate']*100:.2f}%",
            f"- **Khoảng tin cậy 95% (Wilson Score CI):** [{ov['confidence_95_lower']*100:.2f}% - {ov['confidence_95_upper']*100:.2f}%]",
            f"- **Biên độ sai số (Margin of Error):** ±{ov['margin_of_error']*100:.2f}%",
            "",
            "## 2. Bảng Thống Kê Chi Tiết Từng Điểm Dữ Liệu (100 Runs/Điểm)",
            "| Điểm Dữ Liệu | Số Chặng | Thành Công | Độ Tin Cậy (95% CI) | Vận Tốc TB (km/h) | Vượt Xe | Vi Phạm Đèn |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        for bp in data["batch_data_points"]:
            lines.append(
                f"| **{bp['batch_name']}** | {bp['episodes_count']} | {bp['successes']} | "
                f"{bp['success_rate']*100:.1f}% [{bp['ci_lower']*100:.1f}% - {bp['ci_upper']*100:.1f}%] | "
                f"{bp['mean_speed']:.1f} ± {bp['std_speed']:.1f} | {bp['total_overtakes']} | {bp['total_violations']} |"
            )

        lines.extend([
            "",
            "## 3. Biểu Đồ Trực Quan",
            f"![Biểu đồ tiến trình huấn luyện]({os.path.basename(chart_path)})",
            "",
            "---",
            "*Hệ thống tự động kết xuất dữ liệu đồ thị và chỉ số sau khi hoàn thành huấn luyện.*"
        ])
        return "\n".join(lines)
