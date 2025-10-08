import streamlit as st
from groq import Groq
import base64
import io
import cv2
import supervision as sv
from roboflow import Roboflow
from PIL import Image
import tempfile
import os

st.title("DeepMID - 智慧醫學影像診斷系統")

# 初始化 API 金鑰
GROQ_API_KEY = "您的GROQ_API_KEY"
ROBOFLOW_API_KEY = "您的ROBOFLOW_API_KEY"


def run_yolo_detection(image_path):
    """執行 YOLO 肺部 X 光病灶檢測"""
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace().project("chest-xray-yolo")
        model = project.version(6).model

        result = model.predict(image_path, confidence=40, overlap=30).json()
        
        if not result["predictions"]:
            return None, "未檢測到明顯病灶"
        
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_inference(result)
        
        # 讀取並標註圖片
        image = cv2.imread(image_path)
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        # 檢測到的病灶類型
        detected_lesions = list(set(labels))
        
        return annotated_image, detected_lesions
        
    except Exception as e:
        return None, f"YOLO 檢測錯誤: {str(e)}"

def encode_image(image):
    """將圖片轉換為 Base64 編碼"""
    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        # 如果是 OpenCV 圖片
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            return base64.b64encode(encoded_image).decode("utf-8")
        return None

def groq_analyze_image(image, prompt):
    """使用 Groq API 分析圖片"""
    try:
        base64_image = encode_image(image)
        if not base64_image:
            return "圖片編碼失敗"
            
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
         
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API 錯誤: {str(e)}"

def generate_medical_advice(detected_lesions):
    """根據檢測到的病灶生成醫療建議提示詞"""
    lesions_text = "、".join(detected_lesions) if detected_lesions else "未發現明顯病灶"
    
    medical_prompt = f"""
    你是一位專業的放射科醫師，請分析這張肺部X光影像。

    檢測到的病灶類型: {lesions_text}

    請用繁體中文提供專業的醫學分析，包含以下內容:

    1. 影像發現: 詳細描述在X光片中觀察到的異常發現
    2. 可能診斷: 根據病灶特徵提出可能的醫學診斷
    3. 嚴重程度: 評估病情的嚴重程度(輕度/中度/重度)
    4. 建議處置:
       - 進一步檢查建議(如: CT掃描、痰液檢查、血液檢查等)
       - 治療建議(藥物治療、手術等)
       - 追蹤建議
    5. 就醫建議: 建議就診的科別及就醫時機

    請以專業、清晰的方式呈現，並註明這只是AI輔助診斷，最終診斷需由專業醫師確認。
    """
    
    return medical_prompt

# 檔案上傳
uploaded_file = st.file_uploader(
    "請上傳肺部X光影像", 
    type=['jpg', 'jpeg', 'png'],
    help="支援格式: JPG, JPEG, PNG"
)

# 選擇分析模式
analysis_mode = st.radio(
    "選擇分析模式：",
    ["智慧醫學診斷", "自訂提示詞分析"],
    help="智慧醫學診斷會自動檢測病灶並提供醫療建議，自訂模式可輸入任意提示詞"
)

if analysis_mode == "自訂提示詞分析":
    prompt = st.text_area(
        "請輸入提示詞", 
        value="請針對此肺部X光影像，使用繁體中文說明你看到了什麼？有哪些異常發現？",
        height=100
    )
else:
    prompt = None

if st.button("開始分析") and uploaded_file is not None:
    # 建立暫存檔案儲存上傳的圖片
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    try:
        # 顯示原始圖片
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始影像")
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.subheader("病灶標示結果")
            
            # 執行 YOLO 檢測
            with st.spinner("正在檢測肺部病灶..."):
                annotated_image, detected_lesions = run_yolo_detection(temp_image_path)
            
            if annotated_image is not None:
                # 將 OpenCV BGR 轉換為 RGB
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
                
                # 顯示檢測到的病灶
                if isinstance(detected_lesions, list):
                    st.info(f"🔍 檢測到的病灶： {', '.join(detected_lesions)}")
                else:
                    st.info(f"🔍 {detected_lesions}")
            else:
                st.warning("⚠️ 無法生成標示影像")
                detected_lesions = "未檢測到明顯病灶"
        
        # 進行智慧分析
        st.subheader("📋 智慧診斷分析")
        
        with st.spinner("正在進行深度醫學分析..."):
            if analysis_mode == "智慧醫學診斷":
                # 自動生成醫療提示詞
                medical_prompt = generate_medical_advice(
                    detected_lesions if isinstance(detected_lesions, list) else []
                )
                
                # 使用標示後的圖片進行分析
                analysis_image = annotated_image if annotated_image is not None else original_image
                result = groq_analyze_image(analysis_image, medical_prompt)
            else:
                # 使用自訂提示詞
                result = groq_analyze_image(original_image, prompt)
        
        # 顯示分析結果
        st.success("🎉 影像分析完成!")
        
        # 美化輸出
        st.markdown("### 分析結果")
        st.markdown("---")
        st.write(result)
        
        # 免責聲明
        st.warning("""
        ⚠️ **重要提醒**: 
        本系統提供的是AI輔助診斷建議，僅供參考使用。實際的醫學診斷必須由合格的專業醫師進行。
        如有健康疑慮，請立即就醫尋求專業醫療協助。
        """)
        
    except Exception as e:
        st.error(f"分析過程中發生錯誤: {str(e)}")
    
    finally:
        # 清理暫存檔案
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

elif uploaded_file is None:
    st.info("👆 請上傳肺部X光影像開始分析")

# 側邊欄資訊
with st.sidebar:
    st.header("ℹ️ 使用說明")
    st.markdown("""
    
    ### 功能特色：
    - 🫁 自動肺部病灶檢測
    - 📊 智慧醫學分析
    - 💊 治療建議提供
    - 🔍 影像標示可視化
    
    ### 支援檢測的病灶：
    - 0: Aortic enlargement(主動脈擴大)
    - 1: Atelectasis(肺塌陷)
    - 3: Cardiomegaly(心臟肥大)
    - 5: ILD(間質性肺病)
    - 6: Infiltration(浸潤)
    - 7: Lung Opacity(肺部混濁)
    - 8: NoduleMass(結節/腫塊)
    - 10: Pleural effusion(胸腔積液)
    - 13: Pulmonary fibrosis(肺纖維化)
    
    ### 使用流程：
    1. 上傳肺部X光影像
    2. 選擇分析模式
    3. 點擊「開始分析」
    4. 查看標示結果與診斷建議
    """)