# AI AE（含 HDR）研究與實驗紀錄

## 專案目標

利用既有的 `input` 與 `pattern` 資料建立 AI Auto Exposure（AI AE），同時支援 SDR 與 HDR 場景。系統需要處理相機曝光命令的延遲；目前已知曝光命令約在送出後第 3 frame 才生效。

本專案所有實驗、程式、設定、產物與說明都應放在此 `AI_AE` 資料夾內。每次修改或實驗完成後，必須在本 README 的「更新紀錄」中以中文記錄結果，包括失敗實驗與尚未確認的假設。

試跑與驗證中間檔一律使用系統暫存目錄，流程結束時自動清除；`outputs/` 與 `labels/` 只保留可重現的正式 baseline、候選版本、人工 draft 與目前使用中的資料產物，不保留 `trial` 或已淘汰的中間副本。Git 會提交程式、設定、labels、權重、CSV／JSON／NPZ 與 HTML 報告；原始 `input/` 及 `outputs/` 下的 JPG／PNG 等圖檔不提交。HTML inference 報告雖含內嵌縮圖，仍作為單一可攜式報告提交。

## 目前狀態（2026-09-28）

- 已參考分享對話〈AI自動曝光預測架構〉。
- 已讀取 `input/pattern_1` 的 3 張 1920×1080、10-bit、Bayer PGM。
- 已完成以 NumPy MLP 為核心的資料讀取、特徵、訓練、推論、controller 與 closed-loop simulator；DNG／RAW／EXR 準備及 HTML 圖片報告另使用 requirements 中的影像套件。
- `examples/` 內的 camera status 與 AE output 仍是早期 smoke-test placeholder；正式結果應讀取 `labels/scene4_*`、`labels/sihdr_*` 與對應 `outputs/*training*`。
- Scene4 的 100 組 SDR 與 static HDR enable 人工標註都已完成；Version 0 SDR baseline 保持不變，HDR 結果另存為 `0-hdr-candidate`。
- SI-HDR 的 181 個場景已完成 SDR 與 static HDR benefit 人工確認；`sihdr-sdr-v0` 保留為 SDR 正式版本，另完成 `sihdr-hdr-v0-candidate`。HDR ratio 與 anchor 仍無 ground truth，因此 ratio loss 保持 masked。

## 模型實際輸入、架構與輸出

每張曝光 frame 會轉成 101 維統計特徵：32 維 normalized luminance histogram、64 維 8×8 平均亮度 grid，以及 5 維 camera state。camera state 包含目前已套用的 EV、快門時間、sensor×ISP gain、3-frame pending exposure queue 的第一個與最後一個 EV；各值會先縮放到適合 MLP 的數值範圍。Scene4 使用 DNG embedded preview series 2 的亮度統計；SI-HDR 使用 EXR radiance 經真實 CR2 stack 校準後模擬出的 linear camera-luma。兩者不是相同 input domain。

模型是單層共享 encoder 的 multi-head MLP：`101 -> Linear(64) -> ReLU`，再分成四個 head：

| Head | 輸出 | 目前意義 |
|---|---|---|
| EV regression | 1 個 linear scalar `target_ev` | 相對 `100 µs × unity sensor gain × unity ISP gain` 的絕對 log2 exposure target；不是 EV correction，也不是直接的 shutter command |
| HDR enable | 1 個 sigmoid probability `hdr_benefit` | 場景靜止且 AE 已接近 SDR target 時，啟用 HDR 是否有幫助；以 validation-only threshold 轉成 On／Off |
| HDR ratio | 2×／4×／8× softmax | 架構已保留，但 Scene4 與 SI-HDR 目前都沒有 ratio ground truth，因此 loss 完全 masked，現階段不可使用此輸出 |
| Confidence | 1 個 sigmoid probability | 學習人工 label confidence；它不是相機安全性或跨 domain 可靠度的證明 |

訓練採 fixed seed 42、full-batch gradient descent、learning rate 0.003，gradient 逐元素 clip 到 `[-5, 5]`。Composite loss 是 `masked EV MSE + 0.2 × masked HDR BCE + 0.2 × masked ratio CE + 0.1 × confidence BCE`；shutter 邊界的 EV、Unknown HDR 和未標註 ratio 都可各自遮罩。所有 split 都以完整 time step／scene 為單位，同一組 15 張曝光不會跨 train、validation、test。

`DelayAwareController` 另外讀取 `target_ev`，以 pending queue 最後一個未來 EV 為基準，每 frame 最多改變 1 EV，並用 HDR on／off 雙 threshold 避免頻繁切換。訓練模型本身不會輸出 HDR 合成影像，也不會自行決定 shutter／gain 的硬體分配。`test_inference_report.html` 顯示的 prediction 圖是 bracket 中最接近 predicted EV 的既有實拍或模擬 frame；它不是模型生成的新影像。`test_hdr_inference_report.html` 只能檢查 static HDR enable decision，不能驗證 HDR fusion 畫質、motion ghosting 或 sensor HDR ratio。

## Version 0 baseline（2026-09-27）

Version 0 定義為：使用 Scene4 的 100 組人工確認 labels、DNG embedded preview series 2 所提取的 101 維特徵、固定 seed 42、64 維 hidden layer、learning rate 0.003，共訓練 1200 epochs 的 NumPy MLP SDR baseline。

- Training：t000–t069，70 time steps／1050 samples。
- Validation：t070–t084，15 time steps／225 samples。
- Test：t085–t099，15 time steps／225 samples。
- 同一 time step 的 15 張曝光不會跨 split；test 未參與模型更新或最佳 epoch 選擇。
- 正式權重為 `outputs/scene4_training/model.npz`；依 validation loss 保存的權重為 `model_best_validation.npz`。本次最佳 validation epoch 正好是 1200，因此兩者相同。
- 可直接開啟 `outputs/scene4_training/report.html` 查看資料切分、最終指標、training／validation loss 曲線與 overfit 判讀；完整逐 epoch 數值保存在 `loss_history.csv`。
- 可開啟 `outputs/scene4_training/test_inference_report.html`，依圖片查看 held-out test inference。每筆會並排顯示原始 input、prediction 對應的最近實拍 bracket 與人工 target bracket，避免把 bracket 中原本偏暗的 input 誤認為 inference output；報告也可只顯示超出 acceptable range 的圖片、篩選 time step或調整最低誤差門檻。
- Version 0 目前沒有觀察到 validation loss 回升：最後 10% epochs 的 validation loss 仍下降 2.36%。若要延長訓練，應另開受控實驗並保留 validation checkpoint；更優先的改善方向是加入更多 Scene，驗證跨場景泛化。

## Scene4 training data 狀態與操作流程（2026-09-27）

### 現況

- `input/Scene4` 已確認有 1500 張連號 DNG，來源是 ICCV 2023 的 4D AE Dataset Scene 4。
- 資料排列為 100 個 time steps，每個 time step 有 15 張不同快門曝光；ISO 固定 100、光圈固定 f/14。
- 15 個快門依序是 15、8、6、4、2、1、1/2、1/4、1/8、1/15、1/30、1/60、1/125、1/250、1/500 秒。
- 每 15 張應視為同一個 training pattern，例如 `1P0A2006.dng` 至 `1P0A2020.dng` 是 `Scene4_t000`。
- 原始影像與 100 個 time steps 的人工 exposure labels 均已完成，正式 training data 已建立。
- Scene4 只有 SDR exposure stacks，沒有 HDR merge output ground truth。static HDR enable 已用人工判斷補上 supervision；HDR ratio 與 anchor 仍沒有可用 ground truth，因此 ratio loss 必須保持 masked。
- 已實際產生 1500 筆 frame metadata、100 張 contact sheets、100 筆低信心自動建議，以及 shape 為 `(1500, 101)` 的 feature cache。
- 人工確認進度為 100/100；已產生正式 labels、三份資料切分 manifests 與模型輸出。

### 一次性準備

先安裝依賴：

```powershell
python -m pip install -r requirements.txt
```

再產生 frame metadata、低信心自動建議、100 張 contact sheets 與 feature cache：

```powershell
python scripts/prepare_scene4.py
```

輸出如下：

```text
labels/scene4_draft/frame_metadata.csv
labels/scene4_draft/scene_annotations.csv
outputs/scene4_labeling/contact_sheets/Scene4_t000.jpg ... Scene4_t099.jpg
outputs/scene4_labeling/features.npz
outputs/scene4_labeling/prepare_report.json
```

feature cache 目前使用 DNG 內嵌的 1024×683 preview 提取 32-bin histogram、8×8 luma grid 與 camera state。這是先讓完整資料流與 baseline 能訓練的版本；日後若部署端直接使用 Bayer/RAW statistics，training input 也必須換成相同 domain 後重新訓練。

### 人工標註

啟動只綁定本機的標註頁面：

```powershell
python scripts/label_scene4.py
```

瀏覽器會開啟 `http://127.0.0.1:8765`。每個 time step 只需選三個值：

1. 可接受的最暗曝光。
2. 最佳曝光。
3. 可接受的最亮曝光。

頁面最初顯示的是 preview brightness heuristic 產生的低信心建議，只用來減少操作，不是 ground truth。每組按下「儲存並前往下一組」後才會標記為 `human_review`，而且每次操作都立即寫回 `labels/scene4_draft/scene_annotations.csv`。

### 產生 training labels

100 組全部人工確認後執行：

```powershell
python scripts/generate_labels.py build `
  --draft labels/scene4_draft `
  --output labels/scene4_generated `
  --config configs/scene4.json
```

這會產生 `frames.csv`、`scenes.csv` 和模型使用的 `training_samples.csv`。每個 time step 的 scene-level target 會套用到該組 15 張不同曝光輸入，因此共有 1500 筆 training samples。

正式資料另外依 time step 輸出三份獨立 manifest，確保同一組 15 張曝光不會跨資料集：

```text
labels/scene4_generated/splits/train.csv       70 time steps / 1050 samples
labels/scene4_generated/splits/validation.csv  15 time steps / 225 samples
labels/scene4_generated/splits/test.csv        15 time steps / 225 samples
labels/scene4_generated/splits/summary.json
```

### 訓練

```powershell
python scripts/train_scene4.py
```

訓練程式預設拒絕尚未人工確認的 auto labels，並按連續時間區間切分資料，避免同一曝光 stack 被隨機分到不同集合：

```text
train:      t000-t069
validation: t070-t084
test:       t085-t099
```

輸出保存在 `outputs/scene4_training/`，包括 `model.npz`、`model_best_validation.npz`、`report.json`、`report.html`、逐 epoch `loss_history.csv` 與逐樣本 `predictions.csv`。每次訓練完成後也會自動產生 `test_inference_report.html`；若該版本有 HDR supervision，另會產生 `test_hdr_inference_report.html`。只有單一 Scene4 時，test 指標只能表示對後段時間的泛化；正式跨場景評估仍應加入其他 Scene，並以完整 Scene 作 train/validation/test 分割。

訓練程式預設會自動產生 test inference 圖片報告。若只需要重新建立圖片頁、不重新訓練，可執行：

```powershell
python scripts/generate_test_inference_report.py
```

報告會嚴格使用 `predictions.csv` 中的 test split，將 225 張 DNG embedded preview 縮圖直接嵌入單一 `test_inference_report.html`，並另外輸出 `test_inference_report_summary.json`。若同目錄存在 `hdr_predictions_at_target_ev.csv`，同一指令也會產生 `test_hdr_inference_report.html` 與摘要 JSON。Version 0 的 test 結果共有 9/225 張超出 acceptable range；這 9 張集中在 t090–t096 的 exposure index 0 或 1，而且 prediction 均高於人工 target。

### Metric-assisted 第二次 label review

為檢查 Version 0 人工 label 是否有系統性曝光偏差，可先計算每張 bracket 的客觀品質指標：

```powershell
python scripts/analyze_scene4_exposure_quality.py
```

每張 preview 會記錄 normalized luminance entropy、RGB channel saturated ratio、dark ratio、mean luma 與以下初始分數：

```text
quality score = entropy - 2 × saturated ratio - dark ratio
```

結果寫入 `labels/scene4_metric_review/`。原始 Version 0 labels 完整保存在 `original_scene_annotations.csv`；新的 `scene_annotations.csv` 會保留原選擇，但全部標為 `metric_review_pending`，不會在未人工確認前進入正式訓練。

啟動第二次人工 review：

```powershell
python scripts/label_scene4.py --draft labels/scene4_metric_review
```

頁面會並排顯示 Version 0 label、metric 建議，以及 15 張曝光各自的 entropy、saturated ratio、dark ratio、mean luma 與 score。可以套用 metric 建議、還原 Version 0 選擇，或自行調整；每組按下儲存後才會標為 `human_review`。

第一輪 metric 分析和 Version 0 人工 preferred 的比較結果為：35/100 組完全相同，65/100 組建議更短、較暗的曝光，沒有任何一組建議更亮，平均相差 2.42 個 exposure indices。原人工 preferred 集中在 index 4–6，metric preferred 集中在 index 8–10。因此目前資料不支持「人工 label 系統性偏暗」；單看 entropy／saturation／darkness 反而會把 target 往暗處移。Metric 結果只能作為第二次人工 review 的提示，不能直接覆蓋 labels。

100 組完成第二次確認後，候選 labels 應輸出到新的目錄，避免覆蓋 Version 0：

```powershell
python scripts/generate_labels.py build `
  --draft labels/scene4_metric_review `
  --output labels/scene4_generated_v1_candidate `
  --config configs/scene4.json

python scripts/train_scene4.py `
  --labels labels/scene4_generated_v1_candidate/training_samples.csv `
  --split-output labels/scene4_generated_v1_candidate/splits `
  --output outputs/scene4_training_v1_candidate
```

目前 metric-assisted SDR review draft 保留在 `labels/scene4_metric_review/`，但不納入後續 labels 或 training，除非明確重新啟用。

### Scene4 HDR enable 人工標註

Scene4 可以用來標註 `HDR benefit if static` 與明確案例的 binary `HDR enable if static`，但因資料集沒有真正的 HDR merge output，這些 labels 只代表「假設場景靜止時，多曝光是否可能比最佳單張 SDR 更有價值」，不能直接證明 HDR 合成畫質、motion ghosting 或 sensor ratio 正確。

先建立不會修改 Version 0 SDR labels 的獨立 draft：

```powershell
python scripts/prepare_scene4_hdr_review.py
```

再啟動 HDR 專用人工標註頁：

```powershell
python scripts/label_scene4.py `
  --draft labels/scene4_hdr_review `
  --mode hdr
```

每個 time step 依 contact sheet 判斷：最佳單張 SDR 是否無法同時保住重要高光與重要暗部，以及較短／較長曝光是否真的能補回有用細節。標註規則固定為：

| HDR benefit | 判斷 | `hdr_enable_if_static` | Training mask |
|---:|---|---:|---:|
| 0 | 完全沒有幫助 | 0 | 1 |
| 0.25 | 幫助很小，不值得切換 | 0 | 1 |
| 0.5 | 主觀或無法確定 | 留白 | 0 |
| 0.75 | 明顯有幫助 | 1 | 1 |
| 1 | 單張 SDR 明顯不足 | 1 | 1 |

不要因太陽、燈泡、反光點等不重要的小面積 clipping 就標 HDR on。若 bracket 內有移動物體導致無法判讀靜態效果，可標 0.5；實際 runtime 還必須另外加入 motion／ghosting gate，不能只看此模型輸出。

`hdr_ratio_class` 與 `hdr_anchor_filename` 目前應留在 Unknown／空白。必須先有相同 ISP pipeline 產生的 2×、4×、8× HDR 候選影像，或真實 sensor HDR captures，才能比較 ratio 與 anchor；否則 generator 會保持 `hdr_ratio_mask=0`。

100 組 HDR review 完成後，先檢查 train／validation／test 各自是否同時包含 HDR on 與 off，再建立獨立候選版本：

```powershell
python scripts/generate_labels.py build `
  --draft labels/scene4_hdr_review `
  --output labels/scene4_generated_hdr_candidate `
  --config configs/scene4.json

python scripts/train_scene4.py `
  --labels labels/scene4_generated_hdr_candidate/training_samples.csv `
  --split-output labels/scene4_generated_hdr_candidate/splits `
  --output outputs/scene4_training_hdr_candidate `
  --version 0-hdr-candidate
```

HDR 標註與候選訓練已完成。100 組中有 HDR On 65 組、Off 35 組、Unknown 0 組；ratio 與 anchor 仍全部留白。既有連續時間 split 都同時包含兩類：

| Split | Time steps | HDR On | HDR Off |
|---|---:|---:|---:|
| Training t000–t069 | 70 | 50 | 20 |
| Validation t070–t084 | 15 | 5 | 10 |
| Test t085–t099 | 15 | 10 | 5 |

候選模型以 validation split 的 balanced accuracy 選擇 threshold，且每個 time step 只取「最接近人工 SDR target EV 的實拍 frame」作為 HDR 判斷操作點。選出的 probability threshold 是 `0.788064`，結果如下：

| Split | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Training | 100% | 100% | 100% | 100% | 100% | 100% |
| Validation | 100% | 100% | 100% | 100% | 100% | 100% |
| Test | 100% | 100% | 100% | 100% | 100% | 100% |

這個 100% 只適用於 AE 已接近 SDR target 的操作點，而且 test 只有 15 個 time steps。若把同一 threshold 套到所有不同亮度的 bracket frames，test accuracy 是 58.67%、balanced accuracy 是 62.33%，代表 HDR probability 對目前曝光仍很敏感。因此 runtime 現階段只能在 AE 收斂附近評估 HDR；不能把此結果解讀為任意曝光、動態場景或跨 Scene 都已可靠。`0.788064` 也是單一 decision threshold，尚未完成 controller 的 on/off hysteresis thresholds 校正。

候選模型的 SDR test MAE 為 0.4856 EV、RMSE 為 0.6241 EV，96.00% 落在人工 acceptable interval；與 Version 0 接近。Composite training loss 由 172.0057 降至 0.3280，validation loss 由 184.1344 降至 0.5243；最佳 validation epoch 仍是 1200，最後 10% epochs 下降 1.86%，目前沒有 validation loss 回升的跡象。

完整結果可開啟 `outputs/scene4_training_hdr_candidate/report.html`。SDR 圖片比較位於 `test_inference_report.html`；HDR 圖片比較位於 `test_hdr_inference_report.html`，會依 held-out test time step 顯示人工 On/Off、模型 probability、threshold、操作 frame 與完整 15 張曝光 bracket。逐 time step 操作點輸出位於 `hdr_predictions_at_target_ev.csv`；所有 samples、time-step mean 診斷及逐 epoch loss 也保存在同一輸出資料夾。Version 0 權重與報告沒有被覆蓋，這個實驗仍命名為 `0-hdr-candidate`，尚未升為 Version 1。

本次 100 組人工 labels 的正式訓練結果：

```text
training loss:   171.8434 -> 0.2220
validation loss: 183.9994 -> 0.3589
best validation epoch/loss: 1200 / 0.3589

train:      MAE 0.2638 EV / RMSE 0.4135 EV / acceptable interval 99.71%
validation: MAE 0.3688 EV / RMSE 0.5545 EV / acceptable interval 99.11%
test:       MAE 0.4858 EV / RMSE 0.6238 EV / acceptable interval 96.00%
```

Loss 定義為 `EV MSE + 0.2 × masked HDR BCE + 0.2 × masked ratio CE + 0.1 × confidence BCE`。Scene4 的 HDR 與 ratio masks 都是 0，所以曲線主要反映 EV regression 與 confidence。Validation loss 在 epoch 1200 達到目前最低點，最後 120 epochs 仍下降 2.36%，尚未觀察到 overfit；但 training／validation 最終仍有 0.1369 的 generalization gap。繼續增加 epochs 可能小幅改善 validation，但不保證改善 test 或跨場景表現。

這些結果使用 DNG embedded preview series 2 的 histogram/luma features。HDR supervised sample 數量為 0，HDR enable/ratio losses 已被 mask；此模型是 Scene4 SDR target baseline。

## SI-HDR 轉換、人工覆核與候選訓練（2026-09-28）

### 資料來源與本機稽核

SI-HDR 官方資料頁為 [Cambridge Apollo Repository](https://www.repository.cam.ac.uk/items/c02ccdde-db20-4acd-8941-7816ef6b7dc7)，專案說明與論文連結位於 [SI-HDR benchmark](https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/)。資料採 CC BY 4.0；若發布衍生資料、模型或報告，必須保留原作者 attribution 與授權資訊。

本機 `input/raw/sihdr` 已確認：

- 181 個 scene、每 scene 5 張 Canon EOS 5D Mark III CR2，共 905 張真實 RAW、約 21.87 GiB。
- 181 張官方融合 HDR reference EXR；`reference.zip` 為 1,491,596,813 bytes，壓縮檔 CRC 檢查通過。
- CR2 的 ISO 與光圈並不固定，因此不能只依快門時間直接視為 Scene4 f/14、ISO 100 曝光。
- 官方 EXR reference 是 radiance reference；由它重新渲染出的 15 段曝光是模擬資料，不是真實 Scene4 DNG，也不是新的真實 Bayer capture。

### 已完成的轉換

`scripts/prepare_sihdr.py` 會先讀取 CR2 EXIF 與 Bayer black／white levels，以 CFA 對應的線性 RGB 權重建立 luma；再依 shutter、ISO、aperture 將 5 張 RAW stack 正規化到 ISO 100、f/14 radiance rate。官方 EXR luminance 會用 RAW stack 的重疊有效區做 robust scale calibration，最後渲染 Scene4 相同的 15 個 shutter：15、8、6、4、2、1、1/2、1/4、1/8、1/15、1/30、1/60、1/125、1/250、1/500 秒。

目前沒有把結果冒充為 full-resolution Bayer 或 DNG。訓練使用的是模擬 linear camera-luma 所提取的 101 維 histogram／8×8 grid／camera-state features；JPEG 只供 contact sheet 與 HTML 人工觀察。執行：

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_sihdr.py
python scripts/build_sihdr_candidate.py
```

主要產物：

```text
outputs/sihdr_prepared/source_raw_manifest.csv       # 905 張真實 CR2 metadata
outputs/sihdr_prepared/scene_audit.csv               # 181 scene 稽核
outputs/sihdr_prepared/simulation_manifest.csv       # 2,715 張模擬曝光 provenance
outputs/sihdr_prepared/features.npz                  # shape (2715, 101)
outputs/sihdr_prepared/previews/                     # HTML/人工覆核用 JPEG
outputs/sihdr_prepared/contact_sheets/               # 181 張 15-exposure contact sheet
labels/sihdr_draft/                                  # radiance heuristic 初稿
labels/sihdr_generated_candidate/split_assignments.csv
```

### Label、邊界遮罩與 split

以下是人工覆核前的 heuristic candidate 紀錄：preferred exposure、acceptable interval 與 HDR benefit 都由 radiance／clipping heuristic 產生，只能作 pipeline candidate。181 個 scene 中，124 個建議 HDR On、21 個 Off、36 個 Unknown；Unknown 會以 `hdr_label_mask=0` 排除 HDR loss，ratio 全部保持 unknown／masked。

有 64 個 scene 的 preferred exposure 落在 15 秒端點，真正最佳曝光可能位於既有 shutter 範圍之外。這些 scene 全部保留給 HDR head，但設 `target_ev_mask=0`，不會把受截斷的 15 秒值教給 EV regression；其餘 117 scene／1,755 samples 才參與曝光 loss。模型的 loss 現在支援 EV、HDR、ratio、confidence 各自 mask。

資料以完整 scene 做 deterministic stratified split，同一 scene 的 15 張曝光不會跨集合：

| Split | Scenes | Samples | HDR On | HDR Off | HDR Unknown |
|---|---:|---:|---:|---:|---:|
| Training | 127 | 1,905 | 86 | 15 | 26 |
| Validation | 27 | 405 | 19 | 3 | 5 |
| Test | 27 | 405 | 19 | 3 | 5 |

SDR 人工覆核已於 2026-09-28 完成 181/181；若要加入 HDR supervision，再執行第二條 HDR 模式命令：

```powershell
python scripts/label_scene4.py `
  --draft labels/sihdr_draft `
  --contact-sheets outputs/sihdr_prepared/contact_sheets `
  --dataset-name SI-HDR `
  --port 8767

python scripts/label_scene4.py `
  --draft labels/sihdr_draft `
  --contact-sheets outputs/sihdr_prepared/contact_sheets `
  --dataset-name SI-HDR `
  --mode hdr `
  --port 8767
```

每次儲存會立即寫回 CSV；HDR 可將不確定案例留為 benefit 0.5／Unknown。未設定 `hdr_reviewed=1` 的 heuristic 欄位會在 build 時自動清空，確保不會混入正式 loss。

### `sihdr-reference-candidate` 訓練結果

候選模型已用固定 seed 42、hidden dimension 64、learning rate 0.003 訓練 1,200 epochs。這是獨立 SI-HDR 實驗，沒有覆蓋或混入 Scene4 Version 0：

```powershell
python scripts/train_scene4.py `
  --features outputs/sihdr_prepared/features.npz `
  --labels labels/sihdr_generated_candidate/training_samples.csv `
  --config configs/sihdr.json `
  --output outputs/sihdr_training_candidate `
  --input outputs/sihdr_prepared/previews `
  --split-output labels/sihdr_generated_candidate/splits `
  --split-assignments labels/sihdr_generated_candidate/split_assignments.csv `
  --dataset-name SI-HDR `
  --version sihdr-reference-candidate `
  --allow-auto-labels
```

Composite training loss 由 99.4552 降到 1.5797，validation loss 由 111.6628 降到 2.2202；最佳 validation epoch 是 1200，最後 10% 只再下降約 0.91%。目前未見 validation loss 回升，但已接近平臺，因此不建議只增加 epochs。EV test 僅計算有 `target_ev_mask=1` 的 17 scene／255 samples：MAE 0.9838 EV、RMSE 1.3286 EV，acceptable interval 16.86%。

HDR 以 validation 選出的 operating-frame threshold 0.842185 評估，test 22 個有 label 的 scene 中只有 8 個正確，accuracy 36.36%、balanced accuracy 49.12%。這代表目前 heuristic HDR label 加上單張曝光輸入還不夠可靠，不能用這個候選模型取代 Scene4 的人工標註模型，也不應直接部署 HDR enable。

報告位於：

- `outputs/sihdr_training_candidate/report.html`：training／validation loss、overfit 與 split 指標。
- `outputs/sihdr_training_candidate/test_inference_report.html`：17 個 EV-supervised test scenes、255 張逐圖比較。
- `outputs/sihdr_training_candidate/test_hdr_inference_report.html`：22 個 HDR-supervised test scenes、完整 15-frame bracket 與錯誤案例。

### `sihdr-sdr-v0` 人工 SDR 正式版本

181 個 SDR scene 已全部人工確認；label 檔 SHA-256 為 `EF259E098DD366F65CE56774B03D7B83F5B5FA492DED11520B0E5D4699FBFA12`。人工覆核將 shutter 邊界場景由 heuristic 的 64 個降為 20 個，正式 EV supervision 因此有 161 scenes／2,415 samples。HDR 人工覆核仍是 0/181，build 時會清空未覆核的 heuristic HDR 欄位；正式 labels 的 `hdr_label_mask` 與 `hdr_ratio_mask` 均為 0。

| Split | All scenes | All samples | EV-supervised scenes | EV-supervised samples |
|---|---:|---:|---:|---:|
| Training | 127 | 1,905 | 113 | 1,695 |
| Validation | 27 | 405 | 24 | 360 |
| Test | 27 | 405 | 24 | 360 |

用 validation-only 對照 1,200、2,400 與 4,800 epochs；validation loss 分別為 2.7329、2.4695、2.2372。4,800 epochs 最佳且末段只再下降 0.89%，判定接近平臺，因此正式版本固定為 4,800 epochs，不再繼續單純增加 epochs。對照實驗全在系統暫存目錄執行並已刪除，只保留正式版本：

```powershell
python scripts/build_sihdr_candidate.py `
  --output labels/sihdr_generated_sdr_v0

python scripts/train_scene4.py `
  --features outputs/sihdr_prepared/features.npz `
  --labels labels/sihdr_generated_sdr_v0/training_samples.csv `
  --config configs/sihdr.json `
  --output outputs/sihdr_training_sdr_v0 `
  --input outputs/sihdr_prepared/previews `
  --split-output labels/sihdr_generated_sdr_v0/splits `
  --split-assignments labels/sihdr_generated_sdr_v0/split_assignments.csv `
  --dataset-name SI-HDR `
  --version sihdr-sdr-v0 `
  --epochs 4800
```

正式結果：

| Split | MAE (EV) | RMSE (EV) | Acceptable interval |
|---|---:|---:|---:|
| Training | 0.9759 | 1.2738 | 56.46% |
| Validation | 1.1291 | 1.4725 | 50.83% |
| Test | 1.3405 | 1.5883 | 41.67% |

Composite training loss 由 146.3181 降到 1.6915，validation loss 由 140.4724 降到 2.2372；最佳 validation epoch 是 4,800，沒有觀察到 validation loss 回升。報告保存在 `outputs/sihdr_training_sdr_v0/report.html`，逐圖 test 比較為 `test_inference_report.html`。因正式版沒有人工 HDR supervision，所以不產生容易誤導的 HDR test 圖片報告。

### `sihdr-hdr-v0-candidate` 人工 HDR 候選版本

HDR 人工覆核已完成 181/181；更新後完整 annotation CSV 的 SHA-256 為 `3BA0DB53E80EBA756E066AF748D58A97E595F0EAECF0BAE23C8B01FC43105CEA`。分布為 HDR On 101、Off 27、Unknown 53；Unknown 不參與 HDR loss。Ratio 與 anchor 均為 0 筆，因此 `hdr_ratio_mask=0`。明確 HDR supervision 共 128 scenes／1,920 samples。

HDR 與 exposure bin 的 stratified scene split：

| Split | Scenes | HDR On | HDR Off | HDR Unknown |
|---|---:|---:|---:|---:|
| Training | 127 | 71 | 19 | 37 |
| Validation | 27 | 15 | 4 | 8 |
| Test | 27 | 15 | 4 | 8 |

使用 validation composite loss 比較 1,200／2,400／4,800／9,600 epochs，分別為 3.3244／3.0495／2.7148／2.5165；9,600 run 的最佳 validation checkpoint 在 epoch 9,598、loss 2.5160，因此候選版本以 `model_best_validation.npz` 作部署／比較權重。所有 epoch 對照目錄均已刪除。

Validation operating point 選出的 HDR threshold 為 `0.600821`。每個 scene 僅使用最接近人工 SDR target EV 的 frame 評估：

| Split | Scenes | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 90 | 94.44% | 86.84% | 93.42% | 100.00% | 73.68% | 96.60% |
| Validation | 19 | 94.74% | 96.67% | 100.00% | 93.33% | 100.00% | 96.55% |
| Test | 19 | 94.74% | 87.50% | 93.75% | 100.00% | 75.00% | 96.77% |

Test confusion matrix 為 TP 15、TN 3、FP 1、FN 0。若把同一 threshold 套到 test 的所有 285 張 labeled bracket frames，accuracy 為 73.33%、balanced accuracy 73.94%，顯示模型在 AE 接近 target 時較可靠，任意曝光下仍有 domain sensitivity。

同一 candidate 的 SDR test 為 MAE 1.0017 EV、RMSE 1.2825 EV、acceptable interval 52.50%。Composite training loss 由 144.3664 降至 1.6249，validation loss由 145.5376 降至 2.5165；最佳 epoch 接近訓練上限，沒有明顯 overfit 回升，但不再只靠增加 epochs 改善。

正式重現命令：

```powershell
python scripts/build_sihdr_candidate.py `
  --output labels/sihdr_generated_hdr_v0_candidate

python scripts/train_scene4.py `
  --features outputs/sihdr_prepared/features.npz `
  --labels labels/sihdr_generated_hdr_v0_candidate/training_samples.csv `
  --config configs/sihdr.json `
  --output outputs/sihdr_training_hdr_v0_candidate `
  --input outputs/sihdr_prepared/previews `
  --split-output labels/sihdr_generated_hdr_v0_candidate/splits `
  --split-assignments labels/sihdr_generated_hdr_v0_candidate/split_assignments.csv `
  --dataset-name SI-HDR `
  --version sihdr-hdr-v0-candidate `
  --epochs 9600
```

報告位於 `outputs/sihdr_training_hdr_v0_candidate/report.html`；SDR 與 HDR 圖片頁分別為 `test_inference_report.html`、`test_hdr_inference_report.html`。HDR 圖片頁包含 19 個 labeled test scenes 的 operating frame 與完整 15-frame bracket。

這仍是 static HDR enable candidate：SI-HDR 的 15-shutter inputs 是模擬 linear luma，沒有 sensor 真實 HDR output、ratio 畫質、動態 sequence 或 ghosting ground truth。因此不能把 94.74% 解讀為真實相機 HDR pipeline 已可部署。下一步應做 SI-HDR pretraining + Scene4 fine-tuning／domain alignment 對照，並另外取得 motion 與真實 HDR merge 資料。

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

Smoke test 的核心模型只依賴 NumPy；Scene4 DNG preparation 另外使用 Pillow、tifffile 與 imagecodecs，版本需求記錄於 `requirements.txt`。

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

### 2026-09-27｜Scene4 preparation、標註介面與 provisional training

- 依官方資料規格與 DNG EXIF 確認 Scene4 共 1500 張 DNG，排列為 100 time steps × 15 exposures；ISO 100、f/14，快門從 15 秒至 1/500 秒。
- 新增 `scripts/prepare_scene4.py`，會驗證檔名連續性與 EXIF、拆分 `Scene4_t000` 至 `Scene4_t099`、建立 metadata、contact sheets、自動建議與 feature cache。
- 新增 `scripts/label_scene4.py` 本機網頁介面；每組只需確認最暗可接受、最佳與最亮可接受曝光，逐組自動存檔並顯示人工確認進度。
- 新增 `scripts/train_scene4.py`，以連續時間區段切分 train/validation/test，並預設拒絕未經 `human_review` 的 labels。
- Scene4 無 HDR ground truth，因此模型的 HDR enable 與 ratio losses 已接入 masks；測試確認 mask 為 0 時對應 head 權重不會更新。
- 已產生 1500 筆 frame metadata、100 張 contact sheets、100 筆低信心自動建議與 `(1500, 101)` feature cache；目前人工確認進度為 0/100。
- 為驗證 wiring，另以 auto labels 建立 1500 筆 provisional samples 並完成 baseline training：loss 由 152.7224 降至 0.2881，validation MAE 0.4401 EV、test MAE 0.4644 EV。此結果只證明流程可訓練，不代表曝光品質或正式模型表現。
- 單元測試目前 6/6 通過，涵蓋 PGM reader、controller、label generator、Scene4 分組與 HDR mask。
- 下一步：人工確認 100 組 Scene4 labels，生成 `labels/scene4_generated/`，再以不帶 `--allow-auto-labels` 的正式指令訓練。

### 2026-09-27｜Scene4 正式人工 labels 與 training

- 100/100 個 time steps 均已標記為 `human_review`，preferred、acceptable min、acceptable max 無缺值；label generator 的 exposure interval 驗證全部通過。
- 已生成 `labels/scene4_generated/frames.csv`、`scenes.csv` 與 1500 筆 `training_samples.csv`。
- 依完整 time step 做連續時間切分：training 為 t000–t069（70 steps／1050 samples）、validation 為 t070–t084（15 steps／225 samples）、test 為 t085–t099（15 steps／225 samples）。
- 已輸出 `labels/scene4_generated/splits/train.csv`、`validation.csv`、`test.csv` 與 `summary.json`；同一 time step 的 15 張曝光只會出現在同一份資料中。
- 正式模型輸出位於 `outputs/scene4_training/`。loss 從 171.8434 降至 0.2221；validation MAE 為 0.3688 EV，test MAE 為 0.4858 EV。
- validation 有 99.11%、test 有 96.00% 的預測落在人工 acceptable exposure interval 內。由於 acceptable interval 通常涵蓋多個 exposure steps，此比例應搭配 MAE/RMSE 解讀。
- Scene4 沒有 HDR ground truth，HDR supervised samples 為 0，HDR enable/ratio heads 未由本次資料更新。
- 單元測試目前 7/7 通過，新增 train/validation/test 邊界測試。
- 限制：這是單一 Scene 內的時間泛化結果；若要衡量跨場景泛化，需要加入其他 Scene，並以完整 Scene 為單位重新切分。

### 2026-09-27｜Version 0、HTML report 與 loss curve

- 將本次 Scene4 正式模型記為 Version 0：seed 42、hidden dimension 64、learning rate 0.003、1200 epochs，資料 split 與人工 labels 固定不變。
- 訓練程式新增每個 epoch 的 training／validation composite loss 與 EV MSE 紀錄，輸出 `outputs/scene4_training/loss_history.csv`。
- 新增自含式 `outputs/scene4_training/report.html`，可切換 loss 圖的 log／linear Y 軸，並顯示 split、MAE、RMSE、acceptable interval、loss checkpoints 與 overfit 判讀。
- Training loss 由 171.8434 降至 0.2220；validation loss 由 183.9994 降至 0.3589。最低 validation loss 位於 epoch 1200，最後 10% epochs 仍改善 2.36%，目前未觀察到 validation loss 回升。
- 同步輸出 `model_best_validation.npz`。Version 0 的最佳 validation epoch 是最後一個 epoch，因此最佳 checkpoint 與正式 final model 權重相同。
- 是否繼續訓練：可另做較長 epochs 的對照實驗，且必須依 validation loss 保存最佳 checkpoint；目前更需要增加其他 Scene，因為單一 Scene 的額外 epochs 無法證明跨場景泛化會改善。
- 單元測試更新為 8/8 通過，新增 masked Scene4 loss 組成驗證。

### 2026-09-27｜Version 0 test inference 圖片報告

- 新增 `scripts/generate_test_inference_report.py`，讀取 Version 0 的 `predictions.csv`，只選 held-out test split，並解碼 DNG embedded preview series 2。
- 產生單檔 `outputs/scene4_training/test_inference_report.html`，內嵌 225 張 test thumbnails，可依 absolute error、time step、exposure index 排序，或只查看超出 acceptable range 的圖片。
- Test absolute error 的 median 為 0.3955 EV、P90 為 1.0161 EV、最大值為 1.5580 EV；最大 target error 圖片為 `1P0A3305.dng`（Scene4_t086、exposure index 9），但仍落在該組人工 acceptable range 內。
- 實際超出 acceptable range 的圖片共有 9 張，集中在 Scene4_t090–t096 的 exposure index 0／1；這些案例的模型 prediction 全部高於人工 target，顯示最亮輸入端存在一致的正向 exposure bias。
- 另輸出 `test_inference_report_summary.json`，保存 test 統計、最差圖片與各 time step 的 mean／max error，方便後續程式化比較版本。

### 2026-09-27｜Metric-assisted 第二次 label review

- 修正 test inference 圖片報告的視覺語意：每筆改為並排顯示 input、prediction 映射到最近的實拍 bracket，以及人工 target bracket。原報告中的單張圖片只是模型輸入，不能當成模型輸出的曝光影像。
- 新增 `scripts/analyze_scene4_exposure_quality.py`，對 1500 張 preview 計算 luminance entropy、RGB channel saturated ratio、dark ratio、mean luma 與綜合 quality score。
- 建立 `labels/scene4_metric_review/` 作為獨立的第二次 review draft；Version 0 原始 labels 不變，且保存額外快照 `original_scene_annotations.csv`。
- 擴充 `scripts/label_scene4.py`：顯示 Version 0 與 metric 建議、逐曝光 metrics，並提供還原原選擇與套用 metric 建議按鈕。新 draft 目前進度為 0/100，必須逐組儲存才算第二次人工確認。
- 第一輪 metric 建議有 35/100 組與人工 preferred 相同；其餘 65 組全部建議更暗的 exposure index，平均差 2.42 indices。這個結果不支持人工 labels 偏暗，也顯示純 metric 最佳值不能直接視為 ground truth。
- 單元測試更新為 9/9，新增 synthetic bright／normal／dark bracket 測試，確認 heuristic 會排除全飽和與全暗極端。

### 2026-09-27｜Scene4 static HDR enable 標註流程

- Metric-assisted SDR review 保留但暫停使用；其 draft 與分析結果未刪除，也未納入 Version 0 training。
- 新增 `scripts/prepare_scene4_hdr_review.py`，從 Version 0 的 100 組人工 SDR labels 建立獨立 `labels/scene4_hdr_review/`，目前 HDR review 進度為 0/100。
- 擴充本機標註器的 `--mode hdr`，顯示 static HDR benefit、enable、ratio、anchor、confidence 與 notes；HDR 進度和 SDR review 分開計算。
- 固定 benefit／enable 規則：0／0.25 對應 off，0.5 對應 unknown 並 mask，0.75／1 對應 on。前後端與 label generator 都會拒絕不一致的組合。
- Scene4 沒有 HDR merge ground truth，因此 ratio 與 anchor 預設維持 unknown；實際 motion／ghosting safety 必須由未來的動態資料與 runtime gate 處理。
- 單元測試更新為 10/10，新增 HDR draft 保留 SDR label、HDR 儲存與 benefit／enable 一致性驗證。

### 2026-09-27｜Scene4 static HDR labels 完成與候選訓練

- HDR 人工 review 已完成 100/100 組：On 65、Off 35、Unknown 0；ratio 與 anchor 仍保持 unknown，不參與 loss。
- 由原本的完整 time-step 邊界切分建立 `labels/scene4_generated_hdr_candidate/`：training 為 On 50／Off 20，validation 為 On 5／Off 10，test 為 On 10／Off 5。
- 擴充 `scripts/train_scene4.py`，記錄 masked HDR BCE、confusion matrix、precision、recall、specificity、balanced accuracy、F1，並只使用 validation 選擇 HDR threshold。
- 完成獨立的 `0-hdr-candidate` 訓練，輸出至 `outputs/scene4_training_hdr_candidate/`，沒有覆蓋 Version 0。
- 以每組最接近 SDR target EV 的 frame 作為部署操作點時，validation 選出的 threshold 為 0.788064；training／validation／test 的 accuracy 與 balanced accuracy 都是 100%。
- 任意 bracket frame 的 test accuracy 只有 58.67%、balanced accuracy 62.33%，顯示 HDR head 尚未具備曝光不變性。現階段只能在 AE 接近 target 後使用，且仍需要其他 Scene、動態資料、motion gate 及 hysteresis threshold 校正。
- SDR test MAE 0.4856 EV、RMSE 0.6241 EV、acceptable interval 96.00%；composite training loss 由 172.0057 降至 0.3280，validation loss 由 184.1344 降至 0.5243，最佳 epoch 為 1200。
- 新增 `hdr_predictions_at_target_ev.csv` 與 `hdr_predictions_by_time_step.csv`，HTML report 同時呈現操作點、所有曝光 samples 與 diagnostic 結果。
- 單元測試更新為 12/12，新增 HDR binary confusion metrics 與 validation-only threshold selection 測試。

### 2026-09-27｜每次訓練自動產生 SDR／HDR test 圖片報告

- 將 `scripts/generate_test_inference_report.py` 改為支援任意 training output 與版本，不再把頁面固定標為 Version 0。
- `scripts/train_scene4.py` 現在每次訓練結束會自動產生 `test_inference_report.html`；有 HDR supervision 時也會產生 `test_hdr_inference_report.html`，並把兩個連結寫回主 `report.html` 與 `report.json`。
- HDR 頁面只讀取 held-out test split，以每個 time step 最接近 SDR target EV 的 frame 顯示正式 HDR decision；同時列出完整 15 張 bracket，藍框表示操作 frame，紅框表示若在該曝光直接判斷會分類錯誤。
- 已為 `0-hdr-candidate` 補產兩份圖片報告及對應 summary JSON。操作點為 15/15 正確；所有 bracket frames 為 132/225 正確，頁面可依任意曝光錯誤數、probability margin、time step 與人工 label 篩選排序。
- 圖片均為原始 SDR bracket previews，不是 HDR merge output；頁面因此只能檢查 enable decision，不能評估最終 HDR 合成畫質或 ghosting。
- 單元測試更新為 13/13，新增 HDR 圖片報告操作 frame 與完整 bracket 組裝驗證。

### 2026-09-28｜SI-HDR 181-scene 轉換與獨立候選訓練

- 稽核 `input/raw/sihdr`：181 個 scene、905 張 Canon EOS 5D Mark III CR2；下載並驗證官方 181 張 HDR reference EXR，資料授權為 CC BY 4.0。
- 新增 `src/ai_ae/sihdr.py` 與 `scripts/prepare_sihdr.py`，依 RAW black／white level、CFA、shutter、ISO、aperture 合併實拍 stack，校準官方 EXR radiance，模擬 ISO 100、f/14 的 Scene4 15-shutter linear-luma inputs。
- 輸出 2,715 張預覽、181 張 contact sheets、shape `(2715, 101)` feature cache、完整 source/simulation manifests 與 heuristic label draft；所有產物保留 simulated provenance，未宣稱為真實 Bayer／DNG。
- 新增 `scripts/build_sihdr_candidate.py`，建立 deterministic scene-level stratified 127/27/27 split。181 scene 全數保留；64 個 shutter-boundary scene 只遮罩 EV loss，仍可參與 HDR supervision。
- 擴充模型與 trainer 的 `target_ev_mask`，並讓 trainer 支援外部 scene split CSV、dataset 名稱、JPEG inference 圖片。標註器也新增 `--dataset-name`，可直接人工覆核 SI-HDR contact sheets。
- 完成 `sihdr-reference-candidate` 1,200-epoch 獨立訓練：test EV MAE 0.9838 EV、RMSE 1.3286 EV、acceptable 16.86%；HDR operating-point test accuracy 36.36%、balanced accuracy 49.12%。結果顯示管線可運作，但 heuristic labels 與輸入 domain 尚不足以升為正式模型。
- 自動產生 `report.html`、`test_inference_report.html` 與 `test_hdr_inference_report.html`。主報告含 1,201 個 epoch checkpoints；SDR test 頁含 255 張 supervised images，HDR test 頁含 22 個有 label 的 scene 與完整 bracket。
- 編譯檢查與 19/19 單元測試通過。下一步是人工覆核 181 scene，再比較 SI-HDR pretraining + Scene4 fine-tuning 或 domain-aware mixed training；Version 0 未被覆蓋。
- 清除既有 `sihdr_trial`、舊版 usable/candidate draft 與重複 feature cache；候選 builder 改用系統暫存目錄，之後不再保留試跑或中間 draft 產物。

### 2026-09-28｜SI-HDR SDR 181/181 人工確認與正式 Version 0

- 稽核 `labels/sihdr_draft/scene_annotations.csv`：SDR `human_review` 為 181/181、缺欄 0、重複 scene 0；HDR `hdr_reviewed` 仍為 0/181。
- Builder 新增保護：未設定 `hdr_reviewed=1` 的 HDR heuristic 會在 training labels 中清空並 mask；新增回歸測試，完整測試更新為 20/20 通過。
- 人工覆核將 EV shutter-boundary scenes 從 64 降為 20；正式 EV supervision 為 161 scenes／2,415 samples，train／validation／test 分別有 113／24／24 個 EV-supervised scenes。
- 建立 `labels/sihdr_generated_sdr_v0/`，所有 2,715 samples 均來自人工 SDR labels，HDR/ratio supervised samples 均為 0。
- 使用 validation-only 暫存對照選擇 4,800 epochs；1,200／2,400／4,800 的 validation loss 為 2.7329／2.4695／2.2372。暫存對照已全部刪除。
- 完成 `sihdr-sdr-v0` 正式訓練：test MAE 1.3405 EV、RMSE 1.5883 EV、acceptable interval 41.67%；training／validation loss 最終為 1.6915／2.2372，未見 overfit 回升。
- 正式輸出位於 `outputs/sihdr_training_sdr_v0/`，包含 model、validation checkpoint、loss CSV、主 HTML 與 SDR test 圖片 HTML。因 HDR 尚未人工覆核，沒有建立誤導性的正式 HDR test 報告。

### 2026-09-28｜SI-HDR HDR 181/181 人工確認與候選訓練

- HDR 人工覆核完成 181/181：On 101、Off 27、Unknown 53；benefit/enable 不一致 0。Ratio 與 anchor 全部保持空白。
- 建立 `labels/sihdr_generated_hdr_v0_candidate/`；scene-level stratified split 為 127/27/27，train／validation／test 都包含 On、Off 與 Unknown，無 scene leakage。
- 明確 HDR supervision 為 128 scenes／1,920 samples；53 個 Unknown scenes mask HDR loss，ratio supervision 為 0。
- 使用 validation-only 暫存對照選擇 9,600 epochs；1,200／2,400／4,800／9,600 的 validation loss 為 3.3244／3.0495／2.7148／2.5165，最佳 checkpoint 在 epoch 9,598。全部暫存對照已刪除。
- 完成 `sihdr-hdr-v0-candidate`：validation 選出的 operating threshold 為 0.600821；test 19 scenes 的 accuracy 94.74%、balanced accuracy 87.50%、F1 96.77%，confusion matrix 為 TP 15／TN 3／FP 1／FN 0。
- 任意 test bracket frame 的 accuracy 為 73.33%、balanced accuracy 73.94%，仍顯示曝光 domain sensitivity；此模型只可視為 AE 接近 target 時的 static HDR enable candidate。
- 同步輸出主 loss report、360 張 SDR test 圖片頁，以及 19 個 labeled HDR test scenes 的完整 bracket 圖片頁。完整測試維持 20/20 通過。
