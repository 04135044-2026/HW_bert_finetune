# 中文句子情緒分類

以中文 BERT 微調 8 類情緒分類器，在 Google Colab 完成訓練與推論。最佳模型及預測結果已保存在 `output/`。

## 訓練資料來源

- [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)
- 原始資料：[data.csv](data.csv)，使用 UTF-8 編碼，欄位為 `text`、`emotion`。

## 基礎模型

- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
- 使用 `BertForSequenceClassification` 進行單一情緒分類，最多處理 512 tokens。

## 安裝套件

### 主要套件

在 Google Colab 執行 notebook 的安裝儲存格，或執行：

```python
%pip install "transformers==5.20.0" "huggingface-hub>=1.31,<2.0" "datasets>=3,<5" "accelerate>=1.2,<2" scikit-learn matplotlib pandas
```

PyTorch 使用 Colab 預裝版本。安裝或變更套件後，重新啟動工作階段，再從匯入套件的儲存格繼續。

### 套件版本設定

以下依本專案 notebook 的安裝設定列出；未固定的版本以當次 Colab 環境為準，可用 `%pip list` 查看。

| 套件 | 安裝設定 |
| --- | --- |
| torch | 使用 Colab 預裝版本 |
| transformers | `5.20.0` |
| huggingface-hub | `>=1.31,<2.0` |
| datasets | `>=3,<5` |
| accelerate | `>=1.2,<2` |
| scikit-learn | 未固定版本 |
| matplotlib | 未固定版本，供預測 notebook 使用 |
| pandas | 未固定版本，供預測 notebook 使用 |

### 執行環境

- Google Colab。
- 訓練時啟用 CUDA GPU，並使用 FP16。
- 預測可使用 GPU 或 CPU。

## 說明

本模型用於辨識中文日常對話或短句的情緒，分類結果包含以下 8 類。情緒標籤對照保存在 [output/label_mapping.json](output/label_mapping.json)，模型的 `config.json` 也包含相同對照。

### 情緒類別（共 8 類）

| ID | 情緒類別 | 資料筆數 |
| --- | --- | ---: |
| 0 | 厭惡語調 | 404 |
| 1 | 平淡語氣 | 705 |
| 2 | 悲傷語調 | 486 |
| 3 | 憤怒語調 | 527 |
| 4 | 疑問語調 | 386 |
| 5 | 開心語調 | 592 |
| 6 | 關切語調 | 560 |
| 7 | 驚奇語調 | 499 |

共 4,159 筆資料，以隨機種子 42 依情緒分層切分為訓練集 3,327 筆、驗證集 832 筆。

### 訓練參數

| 項目 | 設定 |
| --- | --- |
| Epochs | 3 |
| 訓練／驗證 batch size | 32 / 32 |
| 梯度累積步數 | 2 |
| Learning rate | `1e-4` |
| Warmup steps | 50 |
| Weight decay | 0.01 |
| 最大序列長度 | 512 tokens，batch 內動態 padding |
| 驗證／儲存間隔 | 每 30 個 optimizer steps |
| 最佳模型 | 依驗證 loss 最低選取 |
| 提前停止 | 連續 3 次驗證 loss 未改善 |
| 隨機種子 | 42 |

訓練結束後會載回最佳 checkpoint，並匯出最佳模型至 `output/`。提交時不需要包含 `checkpoint-*` 資料夾。

### 預測結果

作業指定五句文字的預測如下，信心分數顯示至小數點後四位：

| 文字 | 預測情緒 | 信心分數 |
| --- | --- | ---: |
| 我每天都能跟她一起上學，我好開心！ | 開心語調 | 0.9935 |
| 最好的朋友要離開臺灣了，以後可能不容易再見面... | 悲傷語調 | 0.9800 |
| 我覺得我快不行了... | 悲傷語調 | 0.9831 |
| 剛剛收到研究所錄取的通知書！ | 開心語調 | 0.9918 |
| 今年的冬天好像比較晚來。 | 平淡語氣 | 0.9882 |

信心分數為模型輸出經 softmax 轉換的分數，不代表整體準確率。

## 檔案結構

以下對應目前專案的實際目錄；模型檔案直接放在 `output/` 下。

```text
HW_bert_finetune/
├── output/                    # 微調後的最佳模型與預測結果
│   ├── model.safetensors      # 模型權重（約 409 MB，Git LFS）
│   ├── config.json            # 模型配置
│   ├── label_mapping.json     # 情緒標籤對應表
│   ├── tokenizer.json         # 分詞器（Git LFS）
│   ├── tokenizer_config.json  # 分詞器配置
│   ├── training_args.bin      # 訓練參數（Git LFS）
│   ├── trainer_state.json     # 訓練紀錄
│   └── predictions.csv        # 句子預測結果
├── data/                      # 資料處理產生的檔案
│   └── bert/
│       ├── label_mapping.json # 資料標籤對照
│       └── valid.jsonl        # 驗證資料
├── data.csv                   # 重新訓練使用的原始資料
├── finetune.ipynb             # 模型微調 Notebook
├── predict.ipynb              # 模型預測 Notebook
├── .gitattributes             # Git LFS 設定
└── README.md                  # 專案說明文件
```

`model.safetensors`、`training_args.bin` 及 `tokenizer.json` 使用 Git LFS 儲存。

## 使用方式

### 使用現有模型預測

1. 在 [Google Colab](https://colab.research.google.com/) 開啟 `predict.ipynb`。
2. 安裝套件，依 notebook 說明設定 `PROJECT_DIR`。若使用 Google Drive，設定 `USE_GOOGLE_DRIVE = True` 並掛載 Drive。
3. 將完整的 `output/` 放在 `PROJECT_DIR` 下；模型路徑為 `PROJECT_DIR / "output"`。
4. 從匯入套件儲存格開始往下執行，修改 `TEST_TEXTS` 即可預測自訂句子。
5. 預測結果會顯示在 notebook，並存入 `output/predictions.csv`。

### 重新微調模型

1. 在 Colab 開啟 `finetune.ipynb`，並啟用 GPU。
2. 安裝套件、設定專案目錄，準備 `data.csv`；若目錄中沒有資料檔，可依 notebook 提示上傳。
3. 依序執行資料清理、分層切分、模型初始化與訓練儲存格。
4. 完成後，最佳模型、tokenizer、標籤對照與訓練紀錄會匯出至 `output/`。
5. 暫存空間模式會下載 `bert_artifacts.zip`；Drive 模式則直接儲存在指定資料夾。

兩份 notebook 必須使用相同的 `PROJECT_DIR`。若重新訓練需要保留原模型，先修改 `MODEL_DIR` 指向另一個資料夾。

## 成果

- 最佳模型：[output/](output/)，包含模型權重、設定、分詞器與標籤對照。
- 五句文字的分類結果：[output/predictions.csv](output/predictions.csv)，預測分數直接呈現在上方表格。
- 訓練紀錄：[output/trainer_state.json](output/trainer_state.json)。

1. [訓練模型](https://youtu.be/pGrCpSxtYQM)
2. [預測結果](https://youtu.be/7LBi7Spw5P8)

