# TÀI LIỆU ĐẶC TẢ HỆ THỐNG (SYSTEM SPECIFICATION)
## Dự án: Hệ Thống Xe Tự Lái Thông Minh Kết Hợp Định Tuyến Giao Thông (Smart Local-Global Navigation & Autonomous Driving System)
*Phiên bản tài liệu: 1.0.0*  
*Ngày cập nhật: 27/10/2026*

---

## 1. Mục Đích & Phạm Vi Tài Liệu

Tài liệu này đặc tả chi tiết kiến trúc kỹ thuật, luồng dữ liệu, các giao thức kết nối giữa các module và tham số cấu hình của hệ thống. Đây là căn cứ kỹ thuật để:
- Hướng dẫn lập trình viên phát triển đúng chuẩn thiết kế.
- Đánh giá hiện trạng hệ thống ở từng mốc thời gian (checkpoints).
- Dễ dàng nâng cấp, mở rộng tính năng mới (ví dụ: giao tiếp V2X, thuật toán Multi-Agent RL, bản đồ thế giới mở) mà không phá vỡ cấu trúc cốt lõi.

---

## 2. Kiến Trúc Tổng Thể & Luồng Dữ Liệu (System Architecture)

Hệ thống được thiết kế theo mô hình lai 2 cấp phân tầng (**Hierarchical Hybrid Model**):

```mermaid
flowchart TD
    User([Người dùng / Task Manager]) -->|Điểm A, Điểm B| GlobalLayer[Tầng 1: Global Routing Layer]
    
    subgraph GlobalLayerBlock [Tầng 1: Global Routing Engine]
        MapGraph[(Bản đồ Đồ thị NetworkX)] --> Router[Bộ Tìm Đường A*]
        TrafficMonitor[Giả lập Mật độ Giao thông] -->|Hệ số tắc nghẽn| CostCalc[Hàm Chi phí Động Cost Function]
        CostCalc --> Router
        Router --> WaypointGenerator[Bộ sinh Waypoint]
    end
    
    GlobalLayer --> GlobalLayerBlock
    WaypointGenerator -->|Danh sách Waypoints (x, y)| Coordinator[Bộ Điều Phối Trung Gian Hybrid Coordinator]
    
    subgraph LocalLayerBlock [Tầng 2: Local Driving Simulation]
        Coordinator -->|Target Waypoint hiện tại| Env[Môi trường Gymnasium/Pygame]
        Sensors[Cảm biến Radar & Vận tốc] -->|State / Observation| Agent[Mô hình PPO Agent SB3]
        Agent -->|Action: Steering, Throttle, Brake| Env
        Env -->|Tọa độ mới & Trạng thái xe| Coordinator
        TrafficCars[Xe Giao Thông Khác - NPC] --> Env
    end
    
    Coordinator -->|Yêu cầu Re-routing khi tắc nghẽn| GlobalLayerBlock
    Env --> Renderer[Mô-đun Hiển Thị Pygame GUI / Minimap]
```

---

## 3. Đặc Tả Kỹ Thuật Chi Tiết Từng Module

### 3.1. Tầng 1: Định Tuyến Toàn Cục (Global Routing Layer)

#### A. Cấu trúc Đồ thị Mạng Lưới Đường (Graph Structure)
- **Đồ thị $G = (V, E)$:** Sử dụng đồ thị có hướng hoặc vô hướng được quản lý bởi thư viện `NetworkX`.
- **Node $u \in V$:** Đại diện cho ngã ba, ngã tư hoặc nút chuyển hướng.
  - Thuộc tính Node: `id` (int/str), `pos: (x, y)` (tọa độ không gian 2D).
- **Edge $(u, v) \in E$:** Đại diện cho đoạn đường nối giữa hai nút.
  - Thuộc tính Edge:
    - `length` ($L$): Chiều dài vật lý của đoạn đường (mét).
    - `max_speed` ($v_{max}$): Tốc độ thiết kế tối đa cho phép (m/s hoặc km/h).
    - `lanes` ($N_{lane}$): Số làn xe chạy trên đoạn đường.
    - `density` ($\rho$): Mật độ giao thông thực tế hiện tại ($\rho \in [0.0, 1.0]$, với 0: thông thoáng, 1: tê liệt hoàn toàn).

#### B. Hàm Chi Phí Cạnh Động (Dynamic Edge Cost Function)
Chi phí để di chuyển qua đoạn đường $(u, v)$ được tính toán theo thời gian di chuyển ước tính kết hợp hệ số phạt tắc nghẽn:

$$\text{Cost}(u, v) = \frac{L_{(u, v)}}{v_{max(u, v)}} \times \left(1 + \beta \times \left(\frac{\rho_{(u, v)}}{1 - \rho_{(u, v)} + \epsilon}\right)\right)$$

*Trong đó:*
- $L_{(u, v)} / v_{max(u, v)}$: Thời gian di chuyển tối thiểu theo lý thuyết (Free-flow travel time).
- $\rho_{(u, v)}$: Mật độ phương tiện trên đoạn đường.
- $\beta$: Trọng số phạt kẹt xe (mặc định $\beta = 1.5 - 2.0$).
- $\epsilon = 0.05$: Hằng số chống chia cho 0 khi đường kẹt cứng.

#### C. Thuật Toán Tìm Đường
- Thuật toán chính: **$A^*$ (A-Star Search)** với heuristic khoảng cách Euclid:
  $$h(u, \text{Goal}) = \frac{\text{EuclideanDistance}(pos(u), pos(\text{Goal}))}{\max_{(i, j)} v_{max(i, j)}}$$
- Đầu ra: Chuỗi các điểm mốc tọa độ tuyến tính:
  $$\text{Waypoints} = [(x_0, y_0), (x_1, y_1), \dots, (x_n, y_n)]$$

---

### 3.2. Tầng 2: Điều Khiển Cục Bộ & Học Tăng Cường (Local Control Layer)

#### A. Mô Hình Động Học Xe Tự Lái (Bicycle Kinematic Model)
Phương tiện được mô hình hóa bằng trạng thái vật lý $S = (x, y, \theta, v)$:
- Tọa độ trọng tâm: $(x, y)$
- Góc hướng đầu xe (heading angle): $\theta$
- Vận tốc tức thời: $v$
- Phương trình vi phân chuyển động:
  $$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \frac{v}{L_b} \tan(\delta), \quad \dot{v} = a$$
  *(với $L_b$ là chiều dài cơ sở xe, $\delta$ là góc đánh lái, $a$ là gia tốc)*.

#### B. Không Gian Quan Sát (Observation Space)
Vector trạng thái đầu vào của mạng nơ-ron Agent $O \in \mathbb{R}^{d}$:
1. **Thông tin mục tiêu Waypoint:**
   - Khoảng cách đến Waypoint tiếp theo: $d_{wp}$
   - Sai lệch góc lái đến Waypoint: $\Delta\theta_{wp} = \theta_{target} - \theta$
2. **Thông tin nội tại xe:**
   - Vận tốc hiện tại: $v$
   - Gia tốc hiện tại: $a$
3. **Cảm biến chướng ngại vật (Lidar/Radar ảo):**
   - 8 đến 16 tia quét đa hướng xung quanh xe (khoảng cách tối đa 30m - 50m) đo cự ly tới các xe xung quanh hoặc vỉa hè.

#### C. Không Gian Hành Động (Action Space)
- **Kiểu liên tục (Continuous Action):**
  $$A = [\delta_{steer}, a_{accel}] \in [-1.0, 1.0] \times [-1.0, 1.0]$$
  - $\delta_{steer}$: Chuẩn hóa từ góc bẻ lái tối đa $[-\delta_{max}, +\delta_{max}]$.
  - $a_{accel}$: Giá trị dương biểu thị đạp ga ($[0, a_{max}]$), giá trị âm biểu thị phanh ($[-a_{brake\_max}, 0]$).

#### D. Thiết Kế Hàm Phần Thưởng (Reward Function)
Mỗi bước thời gian $t$, Agent nhận phần thưởng $R_t$:

$$R_t = R_{\text{progress}} + R_{\text{speed}} + R_{\text{lane}} - R_{\text{collision}} - R_{\text{action\_jerk}}$$

- **$R_{\text{progress}}$:** Thưởng cho việc rút ngắn cự ly đến Waypoint đích ($d_{wp}^{t-1} - d_{wp}^{t}$).
- **$R_{\text{speed}}$:** Thưởng khi duy trì tốc độ lý tưởng gần sát $v_{target}$ mà không vi phạm tốc độ giới hạn.
- **$R_{\text{lane}}$:** Phạt nhẹ khi xe bị chệch khỏi tâm làn đường hoặc lệch góc quá lớn.
- **$R_{\text{collision}}$:** Phạt cực nặng (ví dụ: -200) và kết thúc Episode (Done) khi xe chạm vào xe khác hoặc lao ra khỏi đường.
- **$R_{\text{action\_jerk}}$:** Phạt hành động giật lắc (đổi góc lái hoặc đạp ga/phanh quá đột ngột) để xe lái mượt mà.

---

### 3.3. Tầng Điều Phối Tích Hợp (Hybrid Coordinator)

Bộ điều phối trung gian giải quyết bài toán đồng bộ giữa 2 tầng:
1. **Theo dõi Waypoint (Waypoint Tracking):**
   - Đặt ngưỡng bán kính chấp nhận $R_{threshold} = 3.0\text{ m}$.
   - Khi xe tự hành tiến vào phạm vi: $\|(x_{car}, y_{car}) - (x_{wp\_i}, y_{wp\_i})\| \le R_{threshold}$, mục tiêu lập tức chuyển sang $Waypoint_{i+1}$.
2. **Cơ chế Tái Lập Lộ Trình (Dynamic Re-routing):**
   - Định kỳ mỗi $T_{check} = 5\text{ giây}$, kiểm tra mật độ các cạnh nằm phía trước trên lộ trình.
   - Nếu phát hiện mật độ cạnh kế tiếp tăng đột biến ($\rho > 0.8$) do tai nạn hoặc ùn ứ giả lập, Tầng 1 sẽ tính toán lại đường đi từ vị trí nút giao gần nhất để né kẹt xe.

### 3.4. Mô Hình Bản Đồ Đô Thị Thực Tế (Realistic Metropolitan Map)
Hệ thống tích hợp bản đồ đô thị đa dạng cấu trúc (`realistic_city.py`):
- **Cấu trúc nút giao phức tạp:** Ngã ba, ngã tư có đèn tín hiệu, bùng binh trung tâm (Quảng trường), đường chéo cắt và đại lộ vành đai.
- **Biển báo giao thông thực tế:**
  - Biển giới hạn tốc độ: 30 km/h (khu dân cư), 50 km/h (nội thành), 80 km/h (đường cao tốc ngoại thành).
  - Biển cấm đi vào (No-entry): phạt trọng số vô cùng lớn $10^9$ để thuật toán $A^*$ tuyệt đối không đi vào.
  - Đèn tín hiệu giao thông (Traffic Lights): tự động chuyển pha Xanh - Đỏ theo chu kỳ.
### 3.5. Hệ Thống Tín Hiệu Xi-Nhan & Chẩn Đoán Va Chạm (Signals & Telemetry)
- **Tín hiệu Xi-nhan (Turn Signals):**
  - Xe khác (NPC) và Xe tự hành (Ego) đều được trang bị đèn xi-nhan màu vàng cam nhấp nháy ở 4 góc đuôi và đầu xe.
  - Khi xe chuyển làn qua trái để xin vượt: nhấp nháy đèn xi-nhan trái.
  - Khi hoàn thành vượt và trở về làn: nhấp nháy đèn xi-nhan phải.
- **Tuần tra bám đường (Road Bounds Constraint):**
  - Các xe NPC tuần tra theo chuỗi Waypoints của tuyến đường, đảm bảo không bao giờ đi lệch hoặc chạy ra ngoài bản đồ.
- **Chế độ kiểm soát hành trình thủ công:**
  - Khi mở ứng dụng: Xe ở trạng thái chờ (`CONFIGURING`). Người dùng toàn quyền chọn Điểm Bắt Đầu và Đích Đến.
  - Xe chỉ khởi hành khi người dùng nhấn nút `▶ BẮT ĐẦU CHẠY`.
- **Hộp thoại Chẩn đoán Va chạm (Crash Diagnostics Dialog):**
  - Phân tích góc va chạm: Đâm đuôi xe phía trước, va quệt hông xe khi tạt đầu/đổi làn, hoặc va chạm tại giao lộ. Dừng mô phỏng và đếm số lần thử lại (`retry_count`).
- **Bảng Thống kê Hoàn thành Chặng (Journey Completion Summary):**
  - Tốc độ tối đa ($v_{max}$), tốc độ trung bình ($\bar{v}$), số lần vượt xe thành công và số lần thử lại trước khi hoàn thành chặng.

- **Màn hình chính (Pygame Window - 1280x720):**
  - **Khu vực mô phỏng chính (Local Driving View):** Chiếm 70% diện tích màn hình. Hiển thị xe tự hành màu nổi bật (xanh neon/cyan), các xe NPC (xám/đỏ), vạch kẻ đường, tia quét radar và các điểm waypoint phát sáng.
  - **Khu vực bản đồ thu nhỏ (Global Minimap):** Chiếm góc trên bên phải, hiển thị toàn bộ đồ thị mạng lưới đường thành phố, các đoạn đường kẹt xe (màu đỏ/vàng) và tuyến đường tối ưu đã chọn (màu xanh lá).
  - **Bảng điều khiển thông số (HUD Panel):**
    - Vận tốc xe: `Speed (km/h)`
    - Khoảng cách đến đích: `Distance to Goal (m)`
    - Tình trạng lộ trình: `Active Waypoint / Total Waypoints`
    - Tình trạng thuật toán: `PPO Model Status`, `Traffic State`

---

## 5. Kế Hoạch Đánh Giá & Đo Lường (KPIs & Metrics)

| Chỉ số | Cách đo lường | Mục tiêu kỳ vọng |
| :--- | :--- | :--- |
| **Tỷ lệ đến đích thành công (Success Rate)** | Số chuyến đến đích an toàn / Tổng số chuyến thử nghiệm | $\ge 90\%$ |
| **Tỷ lệ va chạm (Collision Rate)** | Số lần va chạm / Tổng số chuyến | $< 5\%$ |
| **Thời gian tiết kiệm qua Routing** | Thời gian xe chạy theo $A^*$ động so với đường ngắn nhất thuần túy | Tiết kiệm từ $15\% - 35\%$ thời gian |
| **Độ mượt điều khiển (Comfort / Jerk)** | Biến thiên gia tốc trung bình $\frac{da}{dt}$ | Hạn chế rung lắc, chuyển làn dứt khoát |
| **Tốc độ khung hình (FPS)** | Tần số render giao diện trên máy tính cá nhân | Ổn định $\ge 30 - 60\text{ FPS}$ |

---

## 6. Hướng Nâng Cấp Tương Lai (Roadmap for Upgrades)

1. **Multi-Agent Reinforcement Learning (MARL):** Huấn luyện nhiều xe cùng lúc trong mạng lưới thay vì chỉ 1 xe tự lái giữa các xe NPC di chuyển theo luật cứng.
2. **Giao tiếp V2X (Vehicle-to-Everything):** Xe tự hành nhận tín hiệu đèn giao thông và dữ liệu cảm biến chia sẻ từ xe phía trước.
3. **OpenStreetMap Integration:** Nhập dữ liệu bản đồ đường phố thực tế từ OpenStreetMap (OSM) vào đồ thị NetworkX để mô phỏng một khu vực thành phố thực tế.
