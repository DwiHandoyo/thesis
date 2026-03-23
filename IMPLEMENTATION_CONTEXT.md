# Implementation Context: CQRS Data Synchronization with LQR Control

> Dokumen ini berisi seluruh spesifikasi yang diperlukan untuk mengimplementasikan sistem kontrol sinkronisasi data CQRS.
> Referensi lengkap: `thesis/src/chapters/chapter-3.tex`

---

## 1. Arsitektur Sistem

```
Write Model ──→ Kafka (Message Broker) ──→ Consumer/Message Sink ──→ Elasticsearch (Read DB)
                                                    ↑
                                              [CONTROLLER]
                                           mengatur b dan p
```

- **Write Model**: aplikasi yang menulis data (sumber event)
- **Kafka**: message broker, menyimpan event di topic
- **Consumer/Message Sink**: membaca event dari Kafka, menulis ke Elasticsearch
- **Elasticsearch**: read database (materialized view)
- **Controller**: ditempatkan di consumer, mengatur parameter konsumsi setiap interval kontrol

### Mekanisme Acknowledgement

Consumer menggunakan **manual offset commit** (bukan auto-commit):

1. Consumer poll batch dari Kafka
2. Proses batch → tulis ke Elasticsearch
3. Tunggu Elasticsearch respond "OK" (acknowledge)
4. Baru commit offset ke Kafka

```
enable.auto.commit = false
```

Ini memungkinkan pengukuran latency end-to-end secara akurat.

---

## 2. Variabel Sistem

### State Vector (diobservasi setiap interval kontrol)

```
x = [n, c, m, o, l]^T    (5 dimensi)
```

| Simbol | Variabel | Cara Ukur | Satuan |
|--------|----------|-----------|--------|
| n | Consumer lag (backlog) | Kafka Admin API: `consumer_lag = log_end_offset - committed_offset` | Event (jumlah) |
| c | CPU utilization | Sistem monitoring pada server Elasticsearch | % (0-100) |
| m | Memory utilization | Sistem monitoring pada server Elasticsearch | % (0-100) |
| o | I/O operation | Sistem monitoring pada server Elasticsearch | % atau ops/s |
| l | Latency sinkronisasi | `l = t_ack - t_produce` (timestamp ack dari ES - timestamp produce di Kafka) | ms atau detik |

### Control Vector (output controller setiap interval kontrol)

```
u = [b, p]^T    (2 dimensi)
```

| Simbol | Variabel | Deskripsi | Satuan |
|--------|----------|-----------|--------|
| b | Batch size | Jumlah message yang diambil per poll | Event (jumlah) |
| p | Poll interval | Waktu jeda antar poll | ms |

---

## 3. Lima Metode Kontrol

### 3.1 Kontrol Statik (Baseline)

```python
# Parameter tetap sepanjang eksperimen
b = B_DEFAULT  # dari rata-rata kondisi normal di open-loop
p = P_DEFAULT  # dari rata-rata kondisi normal di open-loop

# Setiap interval kontrol:
def static_control(x):
    return [B_DEFAULT, P_DEFAULT]  # tidak berubah
```

Nilai B_DEFAULT dan P_DEFAULT ditentukan dari rata-rata parameter pada kondisi normal saat eksperimen open-loop.

### 3.2 Kontrol Rule-Based

```python
# Threshold ditentukan dari distribusi state di eksperimen open-loop
BACKLOG_HIGH = ...   # percentile atas dari distribusi backlog
CPU_HIGH = ...       # percentile atas dari distribusi CPU
DELTA_B = ...        # increment/decrement batch size

def rule_based_control(x, b_current, p_current):
    n, c, m, o, l = x
    b = b_current
    p = p_current

    if n > BACKLOG_HIGH:
        b += DELTA_B       # backlog tinggi → perbesar batch
    if c > CPU_HIGH:
        b -= DELTA_B       # CPU tinggi → perkecil batch

    # Clamp ke batas operasional
    b = clamp(b, B_MIN, B_MAX)
    p = clamp(p, P_MIN, P_MAX)
    return [b, p]
```

### 3.3 Kontrol PID

```python
# SISO: input = backlog error, output = batch size
# Poll interval tetap atau diatur terpisah

BACKLOG_REF = ...  # target backlog (idealnya 0 atau mendekati 0)

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def control(self, x):
        n = x[0]  # hanya observasi backlog
        error = BACKLOG_REF - n

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        b = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        b = clamp(b, B_MIN, B_MAX)
        return [b, P_DEFAULT]  # p tetap atau diatur terpisah
```

Tuning Kp, Ki, Kd dilakukan berdasarkan data eksperimen open-loop.

### 3.4 Kontrol LQR (Metode Utama)

```python
import numpy as np
from scipy.linalg import solve_discrete_are

# Model state-space (dari identifikasi sistem open-loop):
# x(t+1) = A @ x(t) + B @ u(t)
A = np.array([...])  # 5x5 matrix (dari eksperimen open-loop)
B = np.array([...])  # 5x2 matrix (dari eksperimen open-loop)

# Matriks pembobot (contoh: konfigurasi seimbang)
Q = np.diag([1, 1, 1, 1, 1])  # 5x5, bobot state
R = np.diag([1, 1])            # 2x2, bobot kontrol

# Solve Riccati equation → gain matrix K
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# Setiap interval kontrol:
def lqr_control(x):
    u = -K @ x
    b = clamp(u[0], B_MIN, B_MAX)
    p = clamp(u[1], P_MIN, P_MAX)
    return [b, p]
```

**Identifikasi matriks A dan B:**
- Jalankan eksperimen open-loop dengan variasi b dan p
- Catat pasangan (x_t, u_t, x_{t+1})
- Gunakan least-squares regression:
  - `[A, B] = argmin ||x_{t+1} - A @ x_t - B @ u_t||^2`

### 3.5 Kontrol ANN

```python
import torch
import torch.nn as nn

class ANNController(nn.Module):
    def __init__(self, state_dim=5, control_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, control_dim),
        )

    def forward(self, x):
        return self.net(x)

# Training:
# - Input: state vectors x dari eksperimen open-loop
# - Target: u*(t) = aksi kontrol optimal (yang menghasilkan J minimum)
# - Loss: MSE antara output jaringan dan u*(t)
# - u*(t) ditentukan dari evaluasi empiris: untuk setiap x,
#   coba beberapa kombinasi (b, p), pilih yang menghasilkan J terendah
```

---

## 4. Fungsi Biaya

```
J = Σ_{t=0}^{T} (x_t^T @ Q @ x_t + u_t^T @ R @ u_t)
```

Digunakan untuk:
- Evaluasi performa semua metode (bukan hanya LQR)
- Sensitivity analysis
- Menentukan u*(t) untuk training ANN

---

## 5. Desain Eksperimen

### 5.1 Tahap 1: Open-Loop (Tanpa Controller)

**Tujuan:** Kumpulkan data untuk identifikasi sistem dan kalibrasi metode.

```
Untuk setiap level beban (rendah, sedang, tinggi):
    Untuk setiap kombinasi (b, p) dalam grid:
        1. Set b dan p secara manual (tetap selama window)
        2. Jalankan selama T_window detik
        3. Catat: x(t) setiap dt detik
        4. Simpan: [(x_t, u_t, x_{t+1}), ...]
```

**Output:**
- Dataset untuk identifikasi A, B (least-squares)
- Distribusi state → threshold untuk rule-based
- Rata-rata parameter normal → default untuk static
- Data training untuk ANN (x → u* mapping)
- Basis tuning Kp, Ki, Kd untuk PID

### 5.2 Tahap 2: Closed-Loop (Dengan Controller)

**Setup:**
- Semua metode dijalankan pada kondisi identik
- Kondisi awal: sistem stabil dengan parameter default
- Interval kontrol: sama untuk semua metode
- Durasi eksperimen: sama untuk semua metode

**4 Skenario Beban:**

```
1. STEP:     Lonjakan beban mendadak
             load(t) = L_low untuk t < t1, L_high untuk t >= t1

2. RAMP:     Peningkatan beban bertahap
             load(t) = L_low + (L_high - L_low) * t / T_ramp

3. IMPULSE:  Lonjakan singkat lalu kembali normal
             load(t) = L_high untuk t1 <= t < t2, L_low lainnya

4. PERIODIC: Beban bolak-balik antara dua level
             load(t) = L_high jika (t mod T_period) < T_period/2, L_low lainnya
```

**Matriks eksperimen:**
```
5 metode × 4 skenario = 20 run eksperimen
```

Setiap run: catat seluruh state, control, dan metrik.

---

## 6. Metrik Evaluasi

### A. Kinerja Sinkronisasi
| Metrik | Formula/Definisi |
|--------|-----------------|
| Backlog rata-rata | `mean(n_t)` selama eksperimen |
| Backlog maksimum | `max(n_t)` selama eksperimen |
| Waktu pemulihan backlog | Waktu dari gangguan sampai `n_t ≈ n_ref` |
| Latency rata-rata | `mean(l_t)` selama eksperimen |
| Latency maksimum | `max(l_t)` selama eksperimen |
| Utilisasi CPU | `mean(c_t)`, `max(c_t)` |
| Utilisasi Memory | `mean(m_t)`, `max(m_t)` |
| Aktivitas I/O | `mean(o_t)`, `max(o_t)` |

### B. Dinamika Kontrol (pada skenario step)
| Metrik | Definisi |
|--------|----------|
| Rise time | Waktu backlog kembali ke referensi setelah step |
| Overshoot | `max(n_t - n_ref) / n_ref × 100%` setelah gangguan |
| Settling time | Waktu backlog masuk dan tetap dalam band toleransi |

### C. Efisiensi Pemrosesan
| Metrik | Formula |
|--------|---------|
| Throughput | Event yang berhasil diproses per detik |
| Efisiensi CPU | `throughput / cpu_util` |
| Efisiensi Memory | `throughput / mem_util` |

### D. Biaya Kontrol
| Metrik | Formula |
|--------|---------|
| J total | `Σ (x^T Q x + u^T R u)` |
| Biaya state | `Σ (x^T Q x)` |
| Biaya aksi | `Σ (u^T R u)` |

---

## 7. Sensitivity Analysis

### 4 Konfigurasi Q (R = I konstan)

```python
# Config 1: Prioritas Backlog
Q1 = np.diag([10, 1, 1, 1, 1])

# Config 2: Prioritas Resource
Q2 = np.diag([1, 10, 10, 10, 1])

# Config 3: Prioritas Latency
Q3 = np.diag([1, 1, 1, 1, 10])

# Config 4: Seimbang
Q4 = np.diag([1, 1, 1, 1, 1])  # = I

# R konstan di semua konfigurasi
R = np.diag([1, 1])  # = I
```

### Evaluasi Adaptabilitas

```python
# Untuk setiap konfigurasi k, hitung J untuk semua 5 metode
# J_i^(k) = total cost function metode i pada konfigurasi k

# Normalized Regret:
def normalized_regret(J_i_k, J_min_k):
    return (J_i_k - J_min_k) / J_min_k

# Mean Regret (metode terbaik overall):
mean_regret_i = mean([normalized_regret(J_i_k, J_min_k) for k in configs])

# Max Regret (metode paling robust):
max_regret_i = max([normalized_regret(J_i_k, J_min_k) for k in configs])
```

**Interpretasi:**
- Mean regret terendah → metode terbaik secara keseluruhan
- Max regret terendah → metode paling robust (tidak pernah buruk)
- Keduanya menunjuk metode yang sama → kesimpulan sangat kuat

---

## 8. Suggested Tech Stack

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| Message Broker | Apache Kafka | Event streaming, consumer lag monitoring |
| Read Database | Elasticsearch | Materialized view, bulk indexing |
| Consumer + Controller | Python atau Go | Poll Kafka, jalankan controller, tulis ke ES |
| Riccati Solver (LQR) | `scipy.linalg.solve_discrete_are` | Hitung gain matrix K |
| ANN Training | PyTorch atau scikit-learn | Train neural network controller |
| System Monitoring | Docker stats / psutil / Prometheus | Ukur CPU, memory, I/O |
| Metrics Collection | Custom logger → CSV/InfluxDB | Catat semua state, control, metrik per interval |
| Visualization | Matplotlib / Grafana | Plot time series, perbandingan metode |

---

## 9. Urutan Implementasi yang Disarankan

```
Phase 1: Setup Infrastructure
├── Kafka + Elasticsearch + Consumer (basic, tanpa controller)
├── Producer (load generator: bisa adjust rate)
├── Metrics collector (catat x, u setiap interval)
└── Verify: consumer bisa baca dari Kafka, tulis ke ES, manual commit

Phase 2: Open-Loop Experiments
├── Grid search: variasi b dan p secara manual
├── 3 level beban: rendah, sedang, tinggi
├── Kumpulkan dataset: [(x_t, u_t, x_{t+1}), ...]
└── Output: CSV dataset untuk semua analisis berikutnya

Phase 3: System Identification
├── Dari dataset open-loop, estimasi A dan B (least-squares)
├── Validasi: prediksi x_{t+1} vs aktual
├── Tentukan threshold untuk rule-based
├── Tentukan default untuk static
└── Tentukan Kp, Ki, Kd untuk PID

Phase 4: Implement Controllers
├── Static controller
├── Rule-based controller
├── PID controller
├── LQR controller (solve Riccati, compute K)
└── ANN controller (train dari open-loop data + u*)

Phase 5: Closed-Loop Experiments
├── 5 metode × 4 skenario beban = 20 runs
├── Catat semua metrik per interval
└── Export data untuk analisis

Phase 6: Analysis
├── Hitung semua metrik per metode per skenario
├── Sensitivity analysis (4 konfigurasi Q)
├── Normalized regret, mean regret, max regret
└── Visualisasi dan kesimpulan
```

---

## 10. Referensi File Thesis

| Konten | Lokasi |
|--------|--------|
| Analisis masalah | `thesis/src/chapters/chapter-3.tex` Section III.1 |
| State-space model | `thesis/src/chapters/chapter-3.tex` Section III.2.1 |
| Fungsi biaya | `thesis/src/chapters/chapter-3.tex` Section III.2.2 |
| LQR | `thesis/src/chapters/chapter-3.tex` Section III.2.3 |
| 5 metode kontrol | `thesis/src/chapters/chapter-3.tex` Section III.2.4 |
| Open-loop design | `thesis/src/chapters/chapter-3.tex` Section III.3.1 |
| Desain eksperimen | `thesis/src/chapters/chapter-3.tex` Section III.4 |
| Skenario beban | `thesis/src/chapters/chapter-3.tex` Section III.4.2 |
| Metrik evaluasi | `thesis/src/chapters/chapter-3.tex` Section III.5 |
| Sensitivity analysis | `thesis/src/chapters/chapter-3.tex` Section III.5.5 |
