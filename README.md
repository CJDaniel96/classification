# Classification

PyTorch 影像分類工具包，涵蓋資料前處理、模型訓練到推論的完整流程。

## 功能特色

- **資料前處理**：從 Pascal VOC 標注自動裁切 ROI，產生分類資料集
- **模型訓練**：支援 MobileNetV3、EfficientNetV2、EfficientViT、ConvNeXt、ResNet（透過 timm）
- **推論**：支援單張圖片、資料夾批次、搭配 YOLO 偵測框三種模式

---

## 安裝

建議使用 [uv](https://docs.astral.sh/uv/) 管理環境：

```bash
uv sync
```

或使用 pip：

```bash
pip install torch torchvision timm tqdm tensorboard pillow
```

---

## 使用流程

### Step 1 — 資料前處理：VOC 標注轉分類資料集 (`voc_to_cls.py`)

從 Pascal VOC XML 標注中，依照指定類別裁切 ROI，輸出可直接供 `ImageFolder` 讀取的分類資料集結構。

```bash
python voc_to_cls.py \
    --images     /path/to/images \
    --annotations /path/to/annotations \
    --output     /path/to/output_dataset \
    --labels     class_a class_b \
    --expand-ratio 1.25 \
    --size       224 \
    --min-size   32
```

**輸出結構**

```
output_dataset/
  class_a/
    img001__class_a__000__12_34_56_78.jpg
    ...
  class_b/
    ...
  metadata.csv          ← 可追溯每張 crop 的來源
```

**參數說明**

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--images` | — | 原始圖片目錄 |
| `--annotations` | — | VOC XML 標注目錄 |
| `--output` | — | 輸出資料集目錄 |
| `--labels` | — | 要裁切的類別名稱（可多個） |
| `--expand-ratio` | `1.25` | BBox 向外擴展比例 |
| `--size` | `224` | 輸出正方形邊長（px） |
| `--min-size` | `32` | 過濾過小物件的最小 BBox 邊長（px） |
| `--pad-color` | `0 0 0` | Padding 顏色（RGB） |
| `--overwrite` | `False` | 覆蓋已存在的輸出目錄 |

---

### Step 2 — 訓練 (`train.py`)

資料集目錄需為 `ImageFolder` 格式（每個類別一個子資料夾）：

```
dataset/
  class_a/
    img1.jpg
  class_b/
    img2.jpg
```

**基本訓練指令**

```bash
python train.py \
    --data-dir  /path/to/dataset \
    --model     efficientnetv2_s \
    --epochs    100 \
    --output-dir runs
```

**常用參數**

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data-dir` | — | 資料集根目錄 *(必填)* |
| `--model` | — | 模型名稱 *(必填，見下方支援清單)* |
| `--epochs` | `50` | 訓練 epoch 數 |
| `--batch-size` | `32` | Batch size；設 `-1` 自動偵測最大值 |
| `--lr` | `1e-3` | 初始學習率 |
| `--optimizer` | `adamw` | 優化器：`adamw` / `adam` / `sgd` |
| `--scheduler` | `cosine` | LR 排程：`cosine` / `step` / `plateau` / `onecycle` / `warmup_cosine` |
| `--val-split` | `0.2` | 驗證集比例 |
| `--patience` | `10` | Early stopping 連續無改善 epoch 數 |
| `--img-size` | `224` | 輸入圖片大小 |
| `--no-pretrain` | `False` | 停用 ImageNet 預訓練權重 |
| `--pretrained-path` | — | 本機預訓練權重路徑（`.pth` / `.pt`） |
| `--no-augment` | `False` | 停用 ColorJitter / Grayscale 增強 |
| `--device` | `auto` | 裝置：`cuda` / `mps` / `cpu` / `auto` |
| `--output-dir` | `runs` | 輸出根目錄 |
| `--resume` | — | 從 checkpoint 恢復訓練 |

**支援的模型**

| 別名 | timm 模型 |
|------|-----------|
| `mobilenetv3` / `mobilenetv3_large` | `mobilenetv3_large_100` |
| `mobilenetv3_small` | `mobilenetv3_small_100` |
| `efficientnetv2` / `efficientnetv2_s` | `efficientnetv2_s` |
| `efficientnetv2_m` / `efficientnetv2_l` | `efficientnetv2_m/l` |
| `efficientvit_m0` ~ `efficientvit_m5` | `efficientvit_m0` ~ `m5` |
| `convnext` / `convnext_tiny` | `convnext_tiny` |
| `convnext_small` / `convnext_base` | `convnext_small/base` |
| `resnet` / `resnet50` | `resnet50` |
| `resnet18` / `resnet34` / `resnet101` / `resnet152` | 對應 ResNet |

timm 支援的任意模型名稱也可直接傳入。

**訓練輸出**

```
runs/
  efficientnetv2_s_20260413_120000/
    checkpoints/
      best.pth    ← 驗證 loss 最低的 checkpoint
      last.pth    ← 最後一個 epoch 的 checkpoint
    tb_logs/      ← TensorBoard logs
    config.json   ← 完整訓練參數紀錄
```

**監控訓練**

```bash
tensorboard --logdir runs
```

**從 Checkpoint 恢復訓練**

```bash
python train.py \
    --data-dir /path/to/dataset \
    --model efficientnetv2_s \
    --resume runs/efficientnetv2_s_20260413_120000/checkpoints/last.pth
```

---

### Step 3 — 推論 (`inference.py`)

三種推論模式，皆支援以下共用參數：

| 參數 | 說明 |
|------|------|
| `--checkpoint` | `best.pth` 或 `last.pth` 路徑 *(必填)* |
| `--topk` | 顯示 Top-K 預測結果（預設 `1`） |
| `--device` | `cuda` / `mps` / `cpu` / `auto` |

#### 模式一：單張圖片

```bash
python inference.py single \
    --checkpoint runs/.../checkpoints/best.pth \
    --image      /path/to/image.jpg \
    --topk       3
```

#### 模式二：資料夾批次推論

```bash
python inference.py batch \
    --checkpoint  runs/.../checkpoints/best.pth \
    --input-dir   /path/to/images \
    --output-csv  result.csv \
    --topk        1
```

#### 模式三：YOLO 偵測框 Crop 分類

讀取 YOLO txt 偵測結果，對每個 BBox crop 後進行分類，適合搭配目標偵測流程使用。

```bash
python inference.py yolo \
    --checkpoint  runs/.../checkpoints/best.pth \
    --image-dir   /path/to/images \
    --label-dir   /path/to/yolo_labels \
    --output-csv  result.csv \
    --save-crops  /path/to/save_crops   # 選填，儲存裁切後的 ROI 圖片
```

YOLO label 格式（每行 `class cx cy w h`，座標為相對值）：

```
0 0.512 0.347 0.102 0.089
1 0.721 0.504 0.085 0.076
```

---

## 常見問題

**Q：推論時出現「未找到類別清單，使用自動命名 class_0 ~ class_N」**

Checkpoint 未包含類別名稱（舊版訓練的 checkpoint）。在 checkpoint 同一目錄下建立 `classes.json`：

```json
["class_a", "class_b"]
```

> ⚠️ 類別順序須與訓練時的資料夾字母排序一致。

新版 `train.py` 已自動將 `class_names` 寫入 checkpoint，重新訓練後即可解決。

---

**Q：訓練時 tqdm 進度條卡住不動，突然跳到 100%**

已修正。原因為 DataLoader worker 每個 epoch 重新啟動的初始化延遲，以及 `logging` 與 `tqdm` 共用 stderr 互相干擾。目前版本已加入 `persistent_workers=True` 及 `logging_redirect_tqdm` 修正此問題。

---

**Q：如何使用自動偵測最大 Batch Size？**

```bash
python train.py --data-dir ... --model ... --batch-size -1
```

會對 GPU 做 forward + backward 壓力測試，自動找出不 OOM 的最大 batch size（CPU 環境固定回傳 8）。
