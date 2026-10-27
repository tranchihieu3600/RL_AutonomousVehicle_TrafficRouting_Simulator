# Hệ Thống Xe Tự Lái Thông Minh Kết Hợp Định Tuyến Giao Thông (Smart Local-Global Navigation & Autonomous Driving System)

Hệ thống mô phỏng (Simulation) tự lái tích hợp kiến trúc phân tầng 2 cấp (**Hybrid Global-Local Architecture**):
1. **Định tuyến toàn cục (Global Routing):** Tìm đường đi tối ưu nhất theo thời gian thực (tránh kẹt xe, tính toán mật độ phương tiện và tốc độ giới hạn) bằng đồ thị `NetworkX` và giải thuật $A^*$/Dijkstra.
2. **Điều khiển cục bộ (Local Driving / Reinforcement Learning):** Xe tự hành nhận các mốc tọa độ (Waypoints) từ tầng toàn cục, tự động giữ làn, bám quỹ đạo, điều chỉnh tốc độ, né tránh chướng ngại vật/va chạm sử dụng học tăng cường sâu (**PPO - Proximal Policy Optimization** với `Stable-Baselines3` / `highway-env` / `Pygame`).

---

## 1. Tính Năng Nổi Bật

- **Bản đồ giao thông dạng đồ thị động:** Biểu diễn các nút giao (Nodes) và tuyến đường (Edges) kèm thông tin chiều dài, số làn, giới hạn tốc độ và mật độ phương tiện thời gian thực.
- **Thuật toán tìm đường thông minh $A^*$ Dynamic Weight:** Trọng số chi phí động kết hợp giữa khoảng cách di chuyển lý tưởng và hệ số tắc nghẽn giao thông.
- **Môi trường mô phỏng cục bộ chi tiết:** Trực quan hóa các làn đường, xe chướng ngại vật ngẫu nhiên, cơ chế vật lý di chuyển của phương tiện (tăng tốc, phanh, bẻ lái).
- **Học tăng cường (Deep RL - PPO):** Huấn luyện mô hình Agent đưa ra quyết định hành động liên tục hoặc rời rạc, tối ưu hóa điểm số thưởng an toàn, đúng lộ trình và thời gian di chuyển.
- **Giao diện trực quan hóa (GUI/Pygame):** Hiển thị đồng thời bản đồ toàn cục (Global Path Map) và góc nhìn thực tế cục bộ (Local Driving Simulation View).
- **Tối ưu hóa chạy Local:** Nhẹ nhàng, mượt mà trên máy tính cá nhân (CPU/GPU thông dụng) mà không đòi hỏi phần cứng đắt đỏ.

---

## 2. Kiến Trúc Hệ Thống (Hybrid Architecture)

```
       [ Người dùng chọn: Điểm A (Start) -> Điểm B (Goal) ]
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ TẦNG 1: ĐỊNH TUYẾN TOÀN CỤC (GLOBAL ROUTING)                 │
  │ • Biểu diễn bản đồ: Graph G = (V, E) qua NetworkX           │
  │ • Chi phí động: Cost = (Length / SpeedLimit) + (Alpha * Jam)│
  │ • Thuật toán tìm kiếm: A* / Dijkstra                        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Xuất danh sách Waypoints
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ TẦNG 2: ĐIỀU KHIỂN CỤC BỘ (LOCAL RL CONTROLLER)             │
  │ • Môi trường: Gymnasium / highway-env / Pygame               │
  │ • Thuật toán học tăng cường: PPO (Stable-Baselines3)        │
  │ • State/Observation: Vị trí so với waypoint, vận tốc, radar │
  │ • Action: Góc bẻ lái, chân ga, chân phanh                   │
  │ • Nhiệm vụ: Bám lộ trình, giữ khoảng cách an toàn, tránh va │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Thư Mục Dự Án

```
RL_AutonomousVehicle_TrafficRouting_Simulator/
├── data/
│   ├── maps/                   # Dữ liệu bản đồ JSON/GraphML
│   └── traffic_profiles/       # Kịch bản mật độ giao thông giả lập
├── models/
│   ├── checkpoints/            # Trọng số lưu tạm trong quá trình train
│   └── saved_models/           # Model PPO tốt nhất đã train xong (.zip)
├── src/
│   ├── __init__.py
│   ├── global_routing/         # Tầng 1: Định tuyến toàn cục
│   │   ├── __init__.py
│   │   ├── graph_manager.py    # Quản lý mạng lưới đồ thị và cập nhật mật độ
│   │   ├── cost_function.py    # Hàm tính chi phí cạnh động
│   │   └── router.py           # Thuật toán tìm đường A* & Dijkstra
│   ├── local_env/              # Tầng 2: Môi trường lái xe cục bộ
│   │   ├── __init__.py
│   │   ├── vehicle.py          # Lớp đối tượng xe tự hành và xe giao thông
│   │   ├── environment.py      # Môi trường Gymnasium tùy chỉnh hoặc highway-env
│   │   └── rewards.py          # Thiết kế hàm Reward / Penalty
│   ├── agent/                  # Huấn luyện và Suy luận AI
│   │   ├── __init__.py
│   │   ├── train_ppo.py        # Pipeline huấn luyện PPO
│   │   └── evaluate.py         # Kiểm thử và đánh giá model
│   ├── integration/            # Kết nối Tầng 1 và Tầng 2
│   │   ├── __init__.py
│   │   └── hybrid_controller.py# Chuyển đổi Waypoint và điều phối hành trình
│   └── visualization/          # Hiển thị trực quan
│       ├── __init__.py
│       ├── pygame_renderer.py  # Render giao diện mô phỏng 2D Pygame
│       └── plot_graph.py       # Vẽ đồ thị bản đồ và mật độ xe
├── tests/                      # Unit test cho các module
├── hethong.md                  # Tài liệu đặc tả kỹ thuật chi tiết của hệ thống
├── todo.txt                    # Danh sách theo dõi tiến độ công việc
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn tổng quan dự án
```

---

## 4. Nền Tảng & Công Nghệ Sử Dụng (Tech Stack)

| Hạng mục | Công nghệ / Thư viện | Vai trò |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python 3.10+ | Ngôn ngữ phát triển cốt lõi |
| **Cấu trúc Đồ thị & Routing** | `NetworkX` | Tạo đồ thị giao thông, tính $A^*$ động |
| **Môi trường Mô phỏng** | `Gymnasium`, `highway-env` | Cung cấp chuẩn giao tiếp RL cho xe tự lái |
| **Học máy & RL** | `PyTorch`, `Stable-Baselines3` | Cài đặt và huấn luyện thuật toán PPO |
| **Tính toán & Dữ liệu** | `NumPy`, `Pandas` | Xử lý ma trận tọa độ, dữ liệu vị trí |
| **Trực quan hóa & Đồ họa** | `Pygame`, `Matplotlib` | Hiển thị mô phỏng 2D và biểu đồ lộ trình |

---

## 5. Hướng Dẫn Cài Đặt & Sử Dụng

### 5.1. Chuẩn bị môi trường

Khuyến nghị sử dụng môi trường ảo Python (`venv` hoặc `conda`):

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows
venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 5.2. Chạy thử nghiệm các module

1. **Kiểm thử tìm đường Tầng 1 (Global Routing):**
   ```bash
   python -m src.global_routing.router
   ```
2. **Chạy huấn luyện AI Tầng 2 (PPO Training):**
   ```bash
   python -m src.agent.train_ppo
   ```
3. **Chạy mô phỏng tích hợp đầy đủ (Hybrid Simulation):**
   ```bash
   python -m src.integration.hybrid_controller
   ```
