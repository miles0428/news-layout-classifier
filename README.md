# News Layout Classifier

YouTube 新聞直播截圖分類器，自動判斷主播位置：

| 類別    | 說明                     |
|---------|--------------------------|
| `left`  | 主播在左，UI 框架在右      |
| `right` | 主播在右，UI 框架在左      |
| `other` | 無固定框架（天氣圖、動態圖形）|

**5x80/20 Cross-Validation 準確率：99.78%**（left: 100%, right: 100%, other: 99.6%）

---

## 安裝

```bash
pip install git+https://github.com/miles0428/news-layout-classifier.git
```

或 clone 後安裝：

```bash
git clone https://github.com/miles0428/news-layout-classifier.git
cd news-layout-classifier
pip install .
```

---

## Python 直接用（推薦）

```python
from news_layout import NewsLayoutClassifier

clf = NewsLayoutClassifier()          # 自動載入預訓練模板，直接可用
label = clf.classify("image.jpg")    # left / right / other
```

進階用法：

```python
clf = NewsLayoutClassifier(pixel_tol=0.06, min_match=0.35)

label, scores = clf.classify_with_scores("image.jpg")
print(scores)  # {'left': 0.87, 'right': 0.12}
```

---

## CLI

```bash
# 分類目錄
news-layout batch INPUT_DIR OUTPUT_DIR

# 單一圖片
news-layout single image.jpg --templates templates.npz

# 評估 benchmark
news-layout bench LEFT_DIR RIGHT_DIR OTHER_DIR --rounds 5 --verbose

# 訓練（自訂資料）
news-layout train left_dir right_dir --out my_templates.npz
```

---

## 演算法

**Pixel-wise Match Ratio（最簡單版本）**

1. 對每個 pixel 計算 `|test - template|`
2. `|diff| < pixel_tol` → 該 pixel 算「匹配」
3. Score = 匹配 pixel 比例（越高越像該類別）
4. 兩個類別的 score 都低於 `min_match` → `other`，否則分數高者勝

**參數：**
- `pixel_tol = 0.05`：每個 pixel 的容忍差異範圍（在 [0,1] 影像空間）
- `min_match = 0.30`：低於此分數即判定為 other
- `active_cols`：只計算邊緣與中央 UI 框架區域（共 31 個欄位）

---

## 預訓練模板

已包在 `src/news_layout/templates/news_layout_templates.npz`，`NewsLayoutClassifier()` 實例化時自動載入。

---

## 專案結構

```
news-layout-classifier/
├── pyproject.toml
├── src/news_layout/
│   ├── __init__.py              # class + 參數
│   ├── __main__.py              # CLI 入口
│   └── templates/
│       └── news_layout_templates.npz  # 預訓練模板
└── README.md
```
