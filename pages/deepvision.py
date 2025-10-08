import streamlit as st
import ollama
from groq import Groq
from PIL import Image
import base64
import io

st.title("DeepVision - AI影像分析助手")  

# 下方填入 Groq API Key
GROQ_API_KEY="您的GROQ_API_KEY"


def encode_image(image):
    """將圖片轉換為 Base64 編碼"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ollama_analyze_image(image, prompt):
    """使用 Ollama API 分析圖片"""
    try:
        base64_image = encode_image(image)
        response = ollama.chat(
            model="llama3.2-vision:11b",
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ollama API 錯誤: {str(e)}"

def groq_analyze_image(image, prompt):
    """使用 Groq API 分析圖片"""
    try:
        base64_image = encode_image(image)
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
            temperature=1,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq API 錯誤: {str(e)}"
  
# API 選擇
api_choice = st.radio(
    "選擇API：", 
    ["Groq(雲端)", "Ollama(本機)"], 
    horizontal=True
)
# 圖檔上傳
uploaded_file = st.file_uploader(
    "上傳圖檔", 
    type=["jpg", "jpeg", "png"], 
    help="請上傳您想分析的圖檔"
)
# 預設提示詞
default_prompt = "請針對此圖片，使用繁體中文說明你看到了什麼？請說明你看到的特點。"
prompt = st.text_area(
    "請輸入提示詞", 
    value=default_prompt, 
    height=100
)
# 分析按鈕
if st.button("分析影像") and uploaded_file is not None:
    column1, column2 = st.columns(2)
    # 讀取上傳的圖檔
    image = Image.open(uploaded_file)
    column1.image(image, caption="已上傳的圖檔",
                  use_container_width=True)
    # 顯示分析中的載入動畫
    with st.spinner(f"使用 {api_choice} API 分析中, 請稍等一下..."):
        # 依據選擇的 API 進行分析
        if api_choice == "Ollama(本機)":
            result = ollama_analyze_image(image, prompt)
        else:
            result = groq_analyze_image(image, prompt)
    # 顯示分析結果
    st.success("影像分析完成!")
    column2.write(result)

# 側邊欄資訊
with st.sidebar:
    st.header("ℹ️ 使用說明")
    st.markdown("""
    
    ### API種類：
    - Groq(雲端)
    - Ollama(本機)：使用llama3.2-vision:11b模型，本機端很給力再選這個，不然會跑無敵久...

    ### 使用流程：
    1. 選擇API
    2. 上傳影像
    3. 輸入提示詞
    4. 點擊「分析影像」
    5. 查看結果
    """)