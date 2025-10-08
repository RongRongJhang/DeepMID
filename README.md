
## 1. Get Started

#### Dependencies and Installation
* Python 3.10.17

1. Clone Repo
```
git clone https://github.com/RongRongJhang/DeepMID.git
```
2. Install Dependencies
```
cd DeepMID
pip install -r requirements.txt
```
3. Run Code
```
python3 runapp.py
```

## 2. 智慧醫學影像診斷系統

本專案旨在結合大型語言模型(LLM)與醫學影像處理技術，開發一個可於瀏覽器端即時運行的智慧診斷平台。本專案使用之主要技術：
- 前端網頁介面：
  採用 Streamlit 建構互動式網頁介面，支援即時影像上傳、結果顯示與診斷報告輸出
- 醫學影像處理：
1. 使用 OpenCV 進行影像的基本操作
2. 使用 Roboflow 線上數據管理平台所提供的 YOLOv11 物件偵測模型 API 進行肺部 X 光影像病灶偵測與標示，能自動辨識如肺塌陷、肺部混濁、胸腔積液等多種類型病灶
- 大型語言模型(LLM)：
  採用 Groq 提供的 Llama 4 Scout 模型 API 進行深度醫學分析，依據檢測結果生成專業報告，內容涵蓋影像觀察、可能診斷、病情評估、治療建議與就醫方向。使用者可選擇「智慧醫學診斷」或「自訂提示詞分析」兩種模式，兼具自動化與彈性應用特性。

![](https://github.com/RongRongJhang/DeepMID/blob/main/DeepMID01.png)
![](https://github.com/RongRongJhang/DeepMID/blob/main/DeepMID02.png)
