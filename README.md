# News Layout Classifier

利用樣板比對（Normalized Cross-Correlation）自動分類新聞直播截圖：

| 類別   | 說明                          |
|--------|-------------------------------|
| `left` | 主播在左，UI 框架在右          |
| `right`| 主播在右，UI 框架在左          |
| `other`| 無固定框架（天氣圖、動態圖形）  |

**5x80/20 Cross-Validation 準確率：97.2%**（left: 100%, right: 100%, other: 95%）

---

## 安裝

```bash
pip install pillow numpy matplotlib
```

## 快速開始

### 1. 訓練模板

```bash
python -m classify.main train \
  /path/to/left_dir \
  /path/to/right_dir \
  /path/to/other_dir \
  --out templates/my_templates.npz
```

### 2. 分類圖片目錄

```bash
python -m classify.main batch \
  /path/to/input_images \
  /path/to/output_dir \
  --templates templates/news_layout_templates.npz
```

### 3. 單一圖片分類

```bash
python -m classify.main single image.jpg \
  --templates templates/news_layout_templates.npz
```

### 4. Cross-Validation Benchmark

```bash
python -m classify.main bench \
  /path/to/left_dir \
  /path/to/right_dir \
  /path/to/other_dir \
  --rounds 5
```

---

## 演算法原理

1. **樣板建構**：對每個類別的所有圖片計算平均圖（mean image），作為類別模板
2. **樣板比對**：新圖片與三個模板個別計算 Normalized Cross-Correlation（NCC）
3. **決策**：
   - NCC 分數最高者為預測類別
   - 若最高分與第二高分差距 < `TOLERANCE_MARGIN`（預設 0.05）→ 判定為 `other`
   - 若最高分 < `MIN_TPL_MATCH`（預設 0.15）→ 判定為 `other`

NCC 分數範圍 [-1, +1]，值越高表示與模板越相似。

---

## 預訓練模板

已附帶 `templates/news_layout_templates.npz`，使用 189 left + 215 right + 521 other 張圖片訓練。

---

## 專案結構

```
news-layout-classifier/
├── classify/
│   ├── __init__.py
│   ├── classifier.py    # 核心分類器
│   └── main.py           # CLI 入口
├── templates/
│   └── news_layout_templates.npz   # 預訓練模板
├── results/              # 分析結果圖
├── requirements.txt
├── .gitignore
└── README.md
```
