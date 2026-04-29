# News Layout Classifier

> **公視手語新聞（PTS Sign Language News）** 畫面分類工具。
> 自動區分三種常見畫面型態。

[YouTube 頻道](https://www.youtube.com/@slnewsptsTaiwan)

---

## 專案目標

公視手語新聞的直播截圖可分為三種畫面：

| 類別     | 說明                                      | 範例                                    |
|----------|-------------------------------------------|----------------------------------------|
| `left`  | 主播在**左**側，AI 螢幕助理（手語翻譯員）在**右**側 | [示範截圖](#left) |
| `right` | 主播在**右**側，AI 螢幕助理（手語翻譯員）在**左**側 | [示範截圖](#right) |
| `other` | 新聞主畫面（主播不在框架內，或無固定版面配置）       | [示範截圖](#other) |

**用途：** 幫助文獻整理、影片分析、或作為後續自動化處理的前處理步驟。

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

## Python 直接用

```python
from news_layout import NewsLayoutClassifier

clf = NewsLayoutClassifier()          # 自動載入預訓練模板，直接可用
label = clf.classify("image.jpg")    # left / right / other
```

---

## CLI

```bash
# 分類整個目錄
news-layout batch INPUT_DIR OUTPUT_DIR

# 分類單一圖片
news-layout single image.jpg

# 測試已分類的資料夾
python test.py ROOT_DIR
```

---

## 演算法

**Pixel-wise Match Ratio**

1. 對每個 pixel 計算 `|test - template|`
2. `|diff| < pixel_tol` → 該 pixel 算「匹配」
3. Score = 匹配 pixel 比例（越高越像該類別）
4. 兩個類別的 score 都低於 `min_match` → `other`，否則分數高者勝

**參數（一般不需要改）：**
- `pixel_tol = 0.05`：每個 pixel 的容忍差異（在 [0,1] 影像空間）
- `min_match = 0.30`：低於此分數即判定為 other
- `active_cols`：只計算邊緣與中央 UI 框架區域（共 31 個欄位）

---

## 專案結構

```
news-layout-classifier/
├── pyproject.toml
├── src/news_layout/
│   ├── __init__.py                    # class
│   ├── __main__.py                    # CLI
│   └── templates/
│       └── news_layout_templates.npz   # 預訓練模板（189 left + 215 right 張圖）
└── README.md
```

---

## 資料來源與模板訓練

預訓練模板使用 [PTS Sign Language News](https://www.youtube.com/@slnewsptsTaiwan) YouTube 直播截圖人工分類後訓練。

若要用自己的資料重新訓練：

```bash
news-layout train left_dir right_dir --out my_templates.npz
```

---

## 免責聲明

本專案為**學習與研究目的**開源。

- 模板與模型僅基於公開 YouTube 截圖資料訓練
- 準確率數據僅代表該特定資料集的表現，不保證在其他資料上的泛化能力
- 本工具不隸屬於、亦不為公視（ PTS ）或其附屬機構立場背書
- 用於學術論文發表、教学或任何非商業用途時，請引用本 Repo：
  > https://github.com/miles0428/news-layout-classifier
