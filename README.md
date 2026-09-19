# AI AE（含 HDR）研究與實驗紀錄

## 專案目標

利用既有的 `input` 與 `pattern` 資料建立 AI Auto Exposure（AI AE），同時支援 SDR 與 HDR 場景。系統需要處理相機曝光命令的延遲；目前已知曝光命令約在送出後第 3 frame 才生效。

本專案所有實驗、程式、設定、產物與說明都應放在此 `AI_AE` 資料夾內。每次修改或實驗完成後，必須在本 README 的「更新紀錄」中以中文記錄結果，包括失敗實驗與尚未確認的假設。

## 目前狀態（2026-09-19）

- 已參考分享對話〈AI自動曝光預測架構〉。
- 已讀取 `input/pattern_1` 的 3 張 1920×1080、10-bit、Bayer PGM。
- 已完成不依賴 PyTorch/Pillow 的 NumPy smoke-test pipeline，涵蓋資料讀取、特徵、訓練、推論、controller 與 closed-loop simulator。
- 目前 camera status 與 AE output 都是 placeholder 範例，只能證明資料流正確，不能作為曝光品質結論。

## 建議的第一版系統

第一版先把「場景判斷」與「硬體控制」分開：

```text
AI 模型
  輸入：histogram、區塊 luma、目前相機狀態（必要時加入影像縮圖）
  輸出：absolute SDR target EV、HDR benefit、HDR ratio class、confidence

Delay-aware controller
  輸入：AI target、目前已套用曝光、pending command queue、硬體限制
  功能：3-frame delay compensation、step limit、HDR hysteresis、shutter/gain 分配
  輸出：合法的 sensor command
```

模型優先預測「絕對目標曝光」而不是每幀累加的 `delta EV`。原因是 AI 會連續數幀看到舊曝光畫面；若重複累加相似的 delta，容易在命令延遲後產生 overshoot。

第一版暫不直接使用 GRU。先以 `MLP + 明確 pending queue/controller` 建立 baseline，再用 closed-loop 指標決定 temporal model 是否真的有幫助。

## 架構、I/O 與 EV 定義（整理版）

### 整體資料流

```text
同一場景目前拍到的 Bayer/統計
        +
該 frame 實際已生效的 exposure time / gain / HDR state
        │
        ▼
Feature extractor
  - histogram
  - 8×8 或其他大小的 luma grid
  - clipping ratio
  - camera state
        │
        ▼
AI scene predictor（第一版可用 MLP）
  ├─ absolute target log exposure
  ├─ HDR benefit if static
  ├─ HDR ratio class
  └─ confidence
        │
        ▼
Deterministic controller
  - 讀取 pending command queue
  - 3-frame delay compensation
  - step limit / hysteresis
  - 將 target exposure 分配成 exposure time + gain
        │
        ▼
Sensor command
```

AI 的責任是回答「這個場景希望到達多少總曝光量」；controller 的責任是回答「考慮延遲與硬體限制，現在應送出哪個 exposure time/gain command」。兩者不應混成同一個 label。

### Input

每一筆模型 input 對應一張實拍 frame，至少包含：

```text
影像特徵：histogram、區塊 luma、highlight/shadow clipping ratio
camera state：該 frame 實際已套用的 exposure time、analog gain、digital gain
模式狀態：SDR/HDR applied mode（若有）
```

模型必須知道該 frame 的實際曝光參數。只看一張偏暗圖片，模型無法分辨是場景本來很暗，還是相機曝光量太低。

第一版的 target predictor 理論上不需要 pending queue，因為同一 scene 的理想 target 不會因 queue 改變；queue 必須提供給 controller。現有 smoke-test 把 queue state 也接進 feature，是為了先驗證所有訊號可以走通，後續正式模型應做「有 queue／無 queue」ablation。

### Output EV 是絕對數值嗎？

是，但這裡的「絕對」是指相對於全資料共用且固定的曝光座標，不是 `current EV + delta EV`，也不必直接等同攝影學的 EV100。

在目前硬體中，sensor gain code 的 32 代表 1×，ISP gain code 的 1024 代表 1×。光圈固定時先定義：

```text
sensor_gain_ratio = sensor_gain_code / 32
isp_gain_ratio = isp_gain_code / 1024

exposure_product = exposure_time_us × sensor_gain_ratio × isp_gain_ratio
log_exposure = log2(exposure_product / reference_exposure_product)
```

`reference_exposure_product` 必須全資料固定。目前程式選定 `100 us × 1× sensor gain × 1× ISP gain` 為 0。如此：

- `100 us × code 32 × code 1024` 對應 0 stop。
- `33000 us × code 512 × code 1024` 相對 reference 為 `log2(330 × 16) ≈ 12.37 stops`。
- `target_log_exposure = 3` 表示目標總曝光量為 reference 的 8 倍。
- 不論目前 frame 是 −2 或 +5，模型對同一靜態 scene 都應輸出相同的 target。
- 模型輸出的 target 不直接指定 shutter/gain 組合；controller 再依 motion blur、noise、sensor range 和量化限制分配。

`exposure_time × sensor_gain_code`（或再乘 ISP gain code）可以作為硬體曝光乘積，但它的範圍很大，且「增加 1」不表示固定亮度差，因此不建議直接作為 regression target。除以固定 unity code 只差一個常數比例；再取 log2 後，每增加 1 才代表曝光量加倍。文件中統一稱為 `log exposure` 或 `exposure stops`，避免與相機測光定義的 EV100 混淆。

若 ISP gain 在資料擷取期間固定為 1024，pattern 只記 sensor gain 與 exposure time 仍足夠，但 metadata 或資料集版本必須註明這個固定值。若 ISP gain 可能改變，就必須逐 frame 一起記錄。若 sensor gain code 並非線性倍率，需用實際 calibration table 換算，不能只除以 32。

### 第一版輸出格式

```yaml
target_log_exposure: float       # 主要 regression output
hdr_benefit_if_static: float     # 0~1
hdr_ratio_class: int             # sensor 支援類別，例如 2/4/8
confidence: float                # 0~1
```

最終 sensor command 則是 controller 的輸出，不是 AI label：

```yaml
exposure_time_us: float
sensor_gain_code: float
isp_gain_code: float
hdr_mode: bool
hdr_ratio: int
effective_frame: int
```

## Pattern 資料如何標註與展開成訓練資料

已確認目前 `input/` 中的 pattern 影像已經過校正處理。這能減少不同 frame 的 pixel-domain 偏差，但不取代逐 frame 的 exposure time、sensor gain 與 ISP gain metadata；模型仍需要這些資料，才能從影像亮度反推場景的曝光需求。

### 建議資料結構

每個 `pattern_xxx` 代表一個不變的靜態場景，資料夾內是不同 exposure time/gain 組合：

```text
input/
├── pattern_0001/
│   ├── frames.txt
│   ├── scene_label.txt
│   ├── frame_000.pgm
│   ├── frame_001.pgm
│   └── ...
├── pattern_0002/
│   └── ...
└── pattern_XXXX/
```

`frames.txt` 必須為每張影像記錄「拍攝時實際生效」的狀態：

```text
filename frame_index exposure_time_us sensor_gain_code isp_gain_code hdr_mode
frame_000.pgm 0 100 32 1024 SDR
frame_001.pgm 1 105 32 1024 SDR
frame_002.pgm 2 110 64 1024 SDR
```

程式可由這些欄位計算每張 frame 的 `applied_log_exposure`。不要只從檔名推測曝光，也不要填入當下送出但尚未生效的 command。

### 每個 scene 要標什麼？

同一個 pattern 只需要一份 scene-level oracle：

```text
target_log_exposure
acceptable_min_log_exposure
acceptable_max_log_exposure
hdr_benefit_if_static
hdr_enable_if_static
hdr_ratio_class
hdr_anchor_log_exposure
label_confidence
label_source
```

最實用的 SDR 標註流程：

1. 將同一 pattern 的曝光 sweep 依 `log_exposure` 排序。
2. 人工選出「可接受」的最低曝光與最高曝光 frame。
3. 將區間中心或偏好的某張 frame 設為 `target_log_exposure`。
4. 記錄 label confidence；若 sweep 太疏或沒有任何可接受 frame，標成低 confidence 或暫不納入訓練。
5. 若相同曝光量有不同 shutter/gain 組合，可另記 noise/blur preference，但第一版不要混入 target exposure label。

例如 pattern_0001 中，−1、0、+1 EV 三張都可接受，而 0 EV 最喜歡：

```text
target_log_exposure = 0.0
acceptable_min_log_exposure = -1.0
acceptable_max_log_exposure = 1.0
```

### 一個 pattern 如何產生多筆 training samples？

假設一個 pattern 有 10 張不同曝光的影像，會展開成 10 筆 input；這 10 筆共用同一份 scene target：

```text
暗曝 frame + 該 frame camera state → 同一 scene target EV
中間 frame + 該 frame camera state → 同一 scene target EV
亮曝 frame + 該 frame camera state → 同一 scene target EV
```

這使模型學會從「觀察到的亮度 + 已知拍攝曝光」推論場景所需的絕對 target，而不是只記住目前影像亮度。

Exposure loss 建議使用 acceptable interval：

```text
prediction < min：loss = distance(prediction, min)
min <= prediction <= max：loss = 0
prediction > max：loss = distance(prediction, max)
```

可再加一個低權重的 target-center Huber loss，讓區間內的輸出仍偏向人工首選 target。

### Train/validation/test 切分

必須以整個 `pattern` 為單位切分。不能將同一場景的暗曝 frame 放進 training、亮曝 frame 放進 validation，否則模型看過同一場景，指標會過度樂觀。

### HDR label 的限制

只有 SDR exposure sweep 時，可以先標「靜態場景是否可能從 HDR 受益」，例如同一曝光無法同時保住高光與暗部。但 HDR ratio 與 anchor 是否真的正確，最好需要：

- sensor 真實 HDR sweep；或
- 通過正式 HDR pipeline 合成的候選結果；或
- 明確且經驗證的 dynamic-range 規則。

沒有上述資料時，建議第一階段只訓練 SDR target，把 HDR label 設為 missing 並 mask HDR loss；不要把未知值標成 HDR off 或 ratio 2。

## 訓練前需要提供的資訊

### A. 一次性硬體／資料集定義

```text
sensor gain unity code（目前為 32）
ISP gain unity code（目前為 1024）
reference exposure time（目前 smoke test 為 100 us）
sensor gain code 是否與真實倍率線性；若否，需要 calibration table
exposure time、gain 的合法範圍與量化 step
ISP gain 是否永遠固定為 1024
HDR 支援的 ratio classes
曝光、HDR mode、ratio 各自的生效延遲
校正處理的內容、版本與所有 pattern 是否一致
```

### B. 每張 frame 的客觀 metadata

```text
scene_id / pattern 名稱
filename
frame_index
實際生效的 exposure_time_us
實際生效的 sensor_gain_code
實際生效的 isp_gain_code（若固定仍建議明記 1024）
applied HDR mode
```

這些欄位不能由影像可靠推測，必須由拍攝系統、log 或資料建立流程提供。

### C. 每個 scene 的人工／規則 annotation

SDR 第一版必要欄位：

```text
preferred_filename
acceptable_min_filename
acceptable_max_filename
label_confidence
label_source
```

HDR 欄位可以先留白：

```text
hdr_benefit_if_static
hdr_enable_if_static
hdr_ratio_class
hdr_anchor_filename
```

留白代表 unknown，生成器會輸出 mask；不要用 0 代表未知。

## 目前 smoke-test 模型實際吃哪些 label？

目前 [model.py](src/ai_ae/model.py) 的訓練參數與 loss 如下：

| Head | 程式目前使用的 label | Loss | 備註 |
|---|---|---|---|
| Target exposure | `target_log_exposure` | MSE | 已使用 |
| HDR | `hdr_enable_if_static` | Binary cross entropy | 已使用；目前沒有 mask |
| HDR ratio | `hdr_ratio_class` | Cross entropy | 已使用；目前沒有 mask |
| Confidence | `label_confidence` | Binary cross entropy | 已使用 |

目前 label 檔中雖然另有下列欄位，但 smoke-test trainer 尚未把它們放進 loss：

```text
acceptable_min/max_log_exposure
hdr_benefit_if_static
hdr_anchor_log_exposure
label_source
hdr_label_mask / hdr_ratio_mask
```

因此目前程式只適合確認 flow。正式訓練前應將 exposure MSE 改成「interval loss + 低權重 preferred target loss」，並讓 HDR losses 套用 mask，避免把沒有 HDR label 的 scene 當成 HDR off。

## Label 生成工具

工具分兩階段，因為 exposure metadata 與人工品質選擇無法安全地只由影像自動猜測。

### 1. 掃描所有 pattern 並建立待填模板

```bash
python3 scripts/generate_labels.py init \
  --input input \
  --draft labels/draft
```

會產生：

```text
labels/draft/frame_metadata.csv
labels/draft/scene_annotations.csv
```

在 `frame_metadata.csv` 填入每張圖的 exposure time 與 gain code；在 `scene_annotations.csv` 選擇 preferred、acceptable min/max frame。HDR 不確定時留白。

### 2. 驗證並生成訓練 labels

```bash
python3 scripts/generate_labels.py build \
  --draft labels/draft \
  --output labels/generated \
  --config configs/smoke_test.json
```

工具會自動：

- 將 gain code 換成倍率。
- 計算 exposure product 與 applied log exposure。
- 由選定的檔名取得 target 與 acceptable interval。
- 檢查 preferred 是否位於 acceptable interval。
- 檢查缺欄、重複 frame、非法 gain、HDR/confidence 範圍。
- 將每個 scene label 展開到該 pattern 的所有 frame。
- 為未知 HDR label 產生 mask。

輸出：

```text
labels/generated/frames.csv           每張 frame 的狀態與 log exposure
labels/generated/scenes.csv           每個 scene 的 oracle label
labels/generated/training_samples.csv 已展開、可供 dataset loader 使用
```

`labels/example_filled/` 是 placeholder 填寫範例，`labels/example_generated/` 是對應輸出，不能當成真實曝光品質標註。

## Label 設計提案

### 1. Scene-level oracle（第一版主要訓練目標）

同一個靜態 scene 的所有 exposure sweep frame 共用：

```yaml
scene_id: string
sdr_target_log_exposure: float
sdr_acceptable_min: float
sdr_acceptable_max: float
hdr_benefit_if_static: float       # 建議連續分數，不只 0/1
hdr_enable_if_static: bool
hdr_ratio_class: string            # 必須依 sensor 支援比例定義
hdr_anchor_log_exposure: float
label_source: string               # 人工、規則、影像品質最佳化或混合
label_confidence: float
label_version: string
```

其中 `sdr_acceptable_min/max` 比單一「完美 EV」更符合曝光主觀性，也能避免模型被迫擬合標註者的微小偏好。可使用 interval loss：預測落在區間內不罰，落在區間外才按距離處罰。

### 2. Frame/rollout state（模擬與評估使用）

```yaml
frame_index: int
applied_log_exposure: float
applied_shutter: float
applied_gain: float
pending_commands: []
applied_hdr_mode: string
pending_hdr_commands: []
oracle_target_log_exposure: float
issued_command: object
command_effective_frame: int
lookup_error_ev: float
```

第一版不建議把 `issued_command` 當成 AI 的主要 label。AI 學 scene target；controller 根據 target、queue 與限制算 command，比直接模仿 command 更容易驗證與除錯。

### 3. HDR label 原則

- 只有靜態資料時，只能可靠標註 `HDR benefit if static`；motion/ghosting risk 必須先 mask，不能把未知當成「無風險」。
- HDR enable 建議由連續 benefit 分數加 controller hysteresis 決定，避免模式反覆切換。
- HDR ratio class、anchor exposure 與 HDR mode 應分開記錄。
- `exposure_delay_frames`、`hdr_mode_delay_frames`、`ratio_delay_frames` 不應預設相同，需以硬體實測為準。

## 資料最小需求

每個靜態 scene 建議至少具有：

```text
scene_id
log_exposure / shutter / analog gain / digital gain
原始或 ISP 後影像
histogram（需知道 bin 數、色域與計算位置）
區塊 luma map（例如 8×8）
highlight clipping ratio
shadow clipping ratio
SDR/HDR mode 與 ratio
```

還需要確認：

1. `input` 與 `pattern` 的檔案位置、格式與對應關係。
2. pattern 是場景 ID、曝光 sweep、測試圖卡，還是期望輸出。
3. 曝光量定義與單位，以及 shutter/gain 的合法範圍與量化方式。
4. 曝光命令是否固定延遲 3 frames；HDR mode 與 ratio 的實際延遲。
5. HDR frame 是實拍、sensor staggered exposure，或可由多張 SDR frame 離線合成。
6. 最終部署算力、延遲、模型大小與可用輸入限制。

## 建議實驗順序

### E0：資料稽核與可視化

解析 input/pattern，建立 scene/frame 索引；檢查缺檔、曝光排序、metadata、動態範圍與資料洩漏。先產出資料統計及每個 scene 的 exposure contact sheet。

### E1：Delay simulator 與 rule baseline

在已知 oracle target 的前提下比較：

- 未考慮 delay 的增量 controller
- absolute target controller
- pending-queue-aware controller

此實驗先證明 simulator 能重現 overshoot，並確認 delay-aware 控制有效，再開始訓練 AI。

### E2：SDR target predictor

以 histogram、luma map、camera state 預測 absolute SDR target，controller 固定負責 delay。資料切分必須以 `scene_id` 為單位，不能把同一 scene 的不同曝光分到 train 與 validation。

### E3：HDR heads

加入 `HDR benefit if static`、ratio class 與 anchor exposure。先只評估靜態 HDR；未有真實 motion sequence 前不宣稱能處理 ghosting。

### E4：Closed-loop randomized rollout

隨機化 initial exposure、pending queue、step limit 與場景切換，評估 settling time、overshoot、oscillation、command variation、亮暗部 clipping 累積量及 HDR switch count。

### E5：Temporal ablation

依序比較：

- MLP + deterministic delay-aware controller
- MLP + explicit pending queue
- GRU/MGU + history
- GRU/MGU + explicit queue

只有 closed-loop 表現有穩定改善且部署成本可接受時，才保留 temporal model。

## 實驗紀錄規則

每次實驗至少記錄：

```text
日期與實驗 ID
目的與假設
資料版本與切分
label 版本
程式與設定檔
random seed
輸出資料夾
主要指標
觀察結果
結論與下一步
```

建議後續目錄結構：

```text
AI_AE/
├── README.md
├── data/                 # 原始資料索引與衍生資料；大檔是否納入版本控制另議
├── configs/
├── src/
├── scripts/
├── tests/
├── experiments/
└── outputs/
```

## 目前已實作的程式

```text
configs/smoke_test.json             smoke test 與 controller 設定
examples/camera_status_example.txt  每張 frame 的相機狀態範例
examples/ae_output_example.txt      scene-level AE/HDR label 範例
src/ai_ae/io.py                     PGM、camera status、label reader
src/ai_ae/features.py               32-bin histogram、8×8 luma、camera/queue state
src/ai_ae/model.py                  NumPy 多頭 MLP 與訓練流程
src/ai_ae/controller.py             absolute target、step limit、HDR hysteresis
scripts/run_smoke_test.py            train → inference → 3-frame-delay rollout
tests/test_core.py                   PGM、範例格式與 controller 測試
outputs/smoke_test/report.json       本次完整機器可讀結果
outputs/smoke_test/model.npz         本次模型權重
```

模型目前具有共享 encoder 及四個 head：

1. absolute SDR target EV（regression）
2. HDR benefit/enable（binary probability）
3. HDR ratio class（2×/4×/8× classification）
4. confidence（binary/continuous confidence）

這個拆法在概念上合理，因為 scene target 與硬體 command 的責任分離；未來即使更換 PyTorch、CNN 或 GRU，controller 與 label 定義仍可保留。

### 執行方式

在本資料夾執行：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_smoke_test.py
```

唯一必要套件為 NumPy；版本需求記錄於 `requirements.txt`。

### Smoke test 的資料假設

- 將 `00000020_bayer.pgm` 示意為 target EV 0 附近的 frame。
- 三張 frame 的 applied EV 暫定為 −4、0、+4 EV。
- target、HDR benefit、ratio 和相機參數全是為了打通流程而填寫的範例，並非從原始檔 metadata 得到。
- 目前直接對 Bayer 數值計算 histogram/luma grid，尚未做 black-level correction、Bayer channel 分離、白平衡或 ISP domain 對齊。正式訓練前必須確認模型實際會取得哪一層的統計。
- lookup 缺少中間曝光時採最近 frame，因此 rollout 的 `lookup_error_ev` 可能很大；這不影響 wiring test，但不能用來評估畫面品質。

## 更新紀錄

### 2026-09-19｜初始化

- 讀取分享對話並整理其核心設計：absolute target、3-frame pending queue、delay-aware controller、SDR/HDR 分開建模、closed-loop rollout 評估。
- 確認目前資料夾尚無可分析的 input、pattern 或程式，因此本次未執行資料實驗，也沒有產生模型結果。
- 建立第一版架構、label schema、資料需求與實驗順序。
- 後續已取得 `input/pattern_1`，相關實作與結果見下一筆紀錄。

### 2026-09-19｜E0-SMOKE-001：最小完整流程

- 輸入：`pattern_1` 三張 10-bit Bayer PGM；檔名為 `00000000`、`00000020`、`00000034`。
- 資料觀察：三張 normalized 前的平均 raw value 約為 1.31、194.92、997.23；分別呈現極暗、中間曝光與大量飽和，可作為流程測試用 sweep。
- 新增 camera status 與 AE output TXT 範例，所有數值明確標示為 placeholder。
- 實作 101 維 feature：32-bin histogram + 8×8 grid + 5 個 camera/queue state。
- 實作共享 encoder、多頭輸出的 NumPy MLP，以及簡化的 multi-task loss/backpropagation。
- 實作 3-frame exposure command queue、absolute target step controller 與 HDR hysteresis。
- 測試結果：3/3 單元測試通過。
- 訓練結果：loss 從 0.45460 降到 0.14599；此數值只檢查 optimizer 與資料流有作用。
- Rollout 結果：由 −4 EV 開始，frame 0 發出的 −3 EV 命令在 frame 3 生效，之後逐步收斂至約 0.008 EV，沒有因看到延遲畫面而重複累加到過曝。
- HDR head 可產生 probability、ratio class，controller 也可套用 on/off hysteresis；但目前只有一個 placeholder HDR label，完全不能衡量 HDR 判斷正確率。
- 結論：最小 AI AE 架構與 closed-loop flow 已可順利執行，absolute target + delay-aware controller 的責任拆分與預期一致。
- 下一步：用真實 camera status 取代範例值；增加不同 scene/pattern 後才進行 scene-level train/validation split 與品質評估。

### 2026-09-19｜文件更新：模型 I/O、EV 與標註流程

- 明確定義 output 為固定 reference 下的 `absolute target log exposure`，不是相對目前 frame 的 delta EV，也避免與 EV100 混用。
- 定義光圈固定時的初版座標：`log2(exposure_time × calibrated total gain / fixed reference)`。
- 明確分離 AI scene predictor 與 deterministic controller：AI 不直接 label shutter/gain command。
- 補充多組 pattern 的建議目錄、逐 frame camera metadata 與逐 scene oracle label。
- 定義同一 pattern 的所有曝光 frame 共用相同 scene target，並要求以 pattern 為單位切分 train/validation/test。
- 建議以 acceptable exposure interval 搭配低權重 target-center loss，降低單點主觀標註的不穩定性。
- 註明只有 SDR sweep 時不應硬填 HDR ratio；未知 HDR label 應 mask loss，待真實或正式合成 HDR 候選資料補齊。

### 2026-09-19｜硬體 gain code 與曝光座標修正

- 依實際規格改用 exposure time 約 100~33000、sensor gain code 32~512、ISP unity code 1024。
- 將總 gain 定義為 `(sensor_gain_code / 32) × (isp_gain_code / 1024)`，避免把 register code 誤當直接倍率。
- 將模型 target 定義為相對 `100 us × 1× × 1×` 的 log2 exposure stops；硬體原始 exposure product 仍可在 controller 內使用。
- 修改 camera status 範例、reader、feature extractor、pipeline 與測試；若 ISP gain 固定，仍在 metadata 中顯式記為 1024。
- 重新執行結果：3/3 測試通過，loss 由 36.86607 降至 0.08386；3-frame rollout 由 0 stop 收斂至約 6.048 stops。
- 結論：模型與 controller 的分層架構不需改變，只需修正 input metadata、gain calibration 與 target 座標。

### 2026-09-19｜GitHub repository 初始化

- 已將 `AI_AE` 初始化為獨立 Git repository，避免誤用家目錄中其他專案的 remote。
- Remote 設為 `https://github.com/morNNii/AI_AE.git`，本機初始 commit 為 `c1b118c`。
- 已加入 `.gitignore`，排除 `.DS_Store`、Python cache、虛擬環境與 log。
- GitHub HTTPS 驗證原先失敗，後續已完成 SSH 設定，並成功將 `main` 推送至 `github.com:morNNii/AI_AE.git`。
- 使用者的全域 Git ignore 另有 `input/` 規則；加入三張 pattern PGM 的權限操作未獲允許，因此目前 GitHub repository 尚不包含範例影像，其餘程式、設定、測試、輸出與文件均已上傳。

### 2026-09-19｜Label 生成流程

- 確認 `input/` 內 pattern 影像已經過校正處理；仍要求保留每張 frame 的實際 exposure time 與 gain metadata。
- 盤點 smoke-test trainer：目前實際使用 target log exposure、HDR enable、HDR ratio class、label confidence；acceptable interval、HDR benefit、HDR anchor 與 masks 尚未接入 loss。
- 新增 `scripts/generate_labels.py`，提供 `init` 掃描模板及 `build` 驗證／生成兩階段流程。
- 新增 `labels/draft/` 待填模板、`labels/example_filled/` placeholder 範例及 `labels/example_generated/` 對應輸出。
- 生成器會換算 gain code、計算 log exposure、用人工選定的 frame 建立 scene oracle、展開 training samples，並為未知 HDR labels 產生 mask。
- 新增 label generator 單元測試；目前全部 4/4 測試通過。
- 下一步：取得真實 frame metadata 與人工 SDR 選圖後生成正式 labels；正式訓練前再將 interval loss 和 HDR mask 接入 trainer。
