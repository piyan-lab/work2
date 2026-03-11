"""
LangChain + Gemini API — Gradio 網頁對話介面 (多對話 + 持久化儲存版本)
支援多個獨立對話紀錄、圖片與 PDF 傳送、角色切換、自動儲存與載入歷史
"""

import sys
import os
import base64
import mimetypes
import uuid
import json
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import gradio as gr

# ── 1. 環境與配置 ────────────────────────────────────────
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ 錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案中有設定。")
    exit(1)

# 初始化 Gemini 模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# 初始角色設定
DEFAULT_PERSONAS = {
    "一般助手": "你是一個親切且專業的 AI 助手，請用繁體中文回答使用者的問題。",
    "專業譯者": "你是一個精通多國語言的翻譯專家，請將使用者的對話精準翻譯成流暢的中文或指定語言。",
    "程式專家": "你是一個經驗豐富的軟體工程師，擅長 Python 與 Web 開發，請提供具備高品質、易讀性的程式碼建議。",
    "冷酷偵探": "你是一個硬漢偵探，說話簡短、精煉，語氣中帶著一絲滄桑與懷疑。"
}

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Web 版人類可讀的對話紀錄資料夾 (與 CLI 版分開)
WEB_HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_history")
os.makedirs(WEB_HISTORY_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
PDF_EXTENSION = ".pdf"
ALL_SUPPORTED = IMAGE_EXTENSIONS | {PDF_EXTENSION}

# ── 2. 序列化與檔案操作 ──────────────────────────────────

def message_to_dict(msg: BaseMessage):
    """將 LangChain 訊息轉換為可 JSON 序列化的字典"""
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "human"
    elif isinstance(msg, AIMessage):
        role = "ai"
    else:
        role = "unknown"
    
    # 處理多模態 content (list) 與純文字 content (str)
    return {"role": role, "content": msg.content}

def dict_to_message(d: dict):
    """將字典還原為 LangChain 訊息物件"""
    role = d.get("role")
    content = d.get("content")
    if role == "system":
        return SystemMessage(content=content)
    elif role == "human":
        return HumanMessage(content=content)
    elif role == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=str(content))

def save_session_to_disk(session):
    """將 Session 儲存為 JSON 檔案 (內部格式 + 人類可讀格式)"""
    # 1. 內部格式 (用於重新載入)
    file_path = os.path.join(HISTORY_DIR, f"{session['id']}.json")
    data = {
        "id": session["id"],
        "name": session["name"],
        "persona": session["persona"],
        "history": session["history"],
        "messages": [message_to_dict(m) for m in session["messages"]]
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 2. 人類可讀格式 (timestamp + role + content)
    readable = []
    for msg in session["messages"]:
        if isinstance(msg, SystemMessage):
            continue
        role = "user" if isinstance(msg, HumanMessage) else "ai"
        content = msg.content
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts) or "[附件]"
        readable.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "content": content
        })
    
    if readable:
        import re
        safe_name = re.sub(r'[\\/:*?"<>|]', '', session["name"]).replace(" ", "_")[:20]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_path = os.path.join(WEB_HISTORY_DIR, f"chat_{ts}_{safe_name}.json")
        with open(readable_path, "w", encoding="utf-8") as f:
            json.dump(readable, f, ensure_ascii=False, indent=2)

def load_all_sessions_from_disk():
    """從硬碟載入所有儲存的 Session"""
    sessions = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(HISTORY_DIR, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["messages"] = [dict_to_message(m) for m in data["messages"]]
                    sessions.append(data)
            except Exception as e:
                print(f"⚠️ 載入檔案 {filename} 失敗: {e}")
    return sessions

def build_multimodal_content(text, file_path):
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "rb") as f:
        file_data = base64.standard_b64encode(f.read()).decode("utf-8")
    mime_type = mimetypes.guess_type(file_path)[0] or ("application/pdf" if ext == ".pdf" else "image/jpeg")
    content = []
    if text: content.append({"type": "text", "text": text})
    content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{file_data}"}})
    return content

# ── 3. 對話管理與邏輯 ────────────────────────────────────

def get_empty_session(persona_name="一般助手", personas_dict=DEFAULT_PERSONAS):
    system_prompt = personas_dict.get(persona_name, DEFAULT_PERSONAS["一般助手"])
    return {
        "id": str(uuid.uuid4()),
        "name": "新對話",
        "persona": persona_name,
        "history": [],
        "messages": [SystemMessage(content=system_prompt)]
    }

def chat_response(message, history, sessions, current_id, persona_name, personas_dict, auto_save):
    user_text = message.get("text", "").strip()
    files = message.get("files", [])

    if not user_text and not files:
        yield history, sessions, gr.update()
        return

    session = next((s for s in sessions if s["id"] == current_id), None)
    if not session:
        yield history, sessions, gr.update()
        return

    # 更新角色
    if session["persona"] != persona_name:
        session["persona"] = persona_name
        system_prompt = personas_dict.get(persona_name, DEFAULT_PERSONAS["一般助手"])
        if len(session["messages"]) > 0 and isinstance(session["messages"][0], SystemMessage):
            session["messages"][0] = SystemMessage(content=system_prompt)

    # 更新標題
    if session["name"] == "新對話":
        session["name"] = (user_text[:15] + "...") if len(user_text) > 15 else (user_text or "媒體對話")

    # 處理訊息輸入
    if files:
        file_path = files[0]
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ALL_SUPPORTED:
            yield history + [{"role": "assistant", "content": f"❌ 不支援格式：{ext}"}], sessions, gr.update()
            return
        prompt = user_text or ("請描述這張圖片" if ext in IMAGE_EXTENSIONS else "請摘要這份文件")
        langchain_msg = HumanMessage(content=build_multimodal_content(prompt, file_path))
        history.append({"role": "user", "content": {"path": file_path}})
        if user_text: history.append({"role": "user", "content": user_text})
    else:
        langchain_msg = HumanMessage(content=user_text)
        history.append({"role": "user", "content": user_text})

    session["messages"].append(langchain_msg)
    
    try:
        response_text = ""
        history.append({"role": "assistant", "content": ""})
        for chunk in llm.stream(session["messages"]):
            response_text += chunk.content
            history[-1]["content"] = response_text
            yield history, sessions, gr.update(choices=[(s["name"], s["id"]) for s in sessions])
        
        session["history"] = history
        session["messages"].append(AIMessage(content=response_text))
        
        if auto_save:
            save_session_to_disk(session)
            
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 錯誤：{e}"})
        yield history, sessions, gr.update()

def create_new_chat(sessions, persona_name, personas_dict):
    new_s = get_empty_session(persona_name, personas_dict)
    sessions.append(new_s)
    choices = [(s["name"], s["id"]) for s in sessions]
    return [], sessions, new_s["id"], gr.update(choices=choices, value=new_s["id"]), persona_name

def switch_chat(selected_id, sessions):
    session = next((s for s in sessions if s["id"] == selected_id), None)
    if session: return session["history"], selected_id, session["persona"]
    return [], selected_id, "一般助手"

def manual_save_all(sessions):
    """手動儲存所有 Session"""
    for s in sessions:
        save_session_to_disk(s)
    return "✅ 已成功儲存所有對話至 history/"

# ── 4. UI 佈局 ──────────────────────────────────────────

css = """
.sidebar { background-color: #f7f7f8; border-right: 1px solid #ddd; padding: 15px; }
.chat-container { padding: 20px; }
"""

with gr.Blocks() as demo:
    # 啟動時從硬碟載入
    initial_sessions = load_all_sessions_from_disk()
    if not initial_sessions: initial_sessions = [get_empty_session()]
    
    sessions_state = gr.State(initial_sessions)
    personas_state = gr.State(DEFAULT_PERSONAS.copy())
    current_session_id = gr.State(initial_sessions[0]["id"])

    with gr.Row():
        # 側邊欄
        with gr.Column(scale=1, elem_classes="sidebar", min_width=280):
            gr.Markdown("## 🗂️ 對話與存檔")
            new_chat_btn = gr.Button("➕ 開啟新對話", variant="primary")
            history_list = gr.Dropdown(
                label="歷史紀錄清單",
                choices=[(s["name"], s["id"]) for s in initial_sessions],
                value=initial_sessions[0]["id"],
                interactive=True
            )
            
            with gr.Row():
                auto_save_cb = gr.Checkbox(label="自動儲存", value=True)
                save_btn = gr.Button("💾 手動存檔", size="sm")
            
            save_status = gr.Markdown("-" if initial_sessions[0]["name"] == "新對話" else "已載入舊對話")

            gr.Markdown("---")
            gr.Markdown("## 🎭 角色切換")
            persona_dropdown = gr.Dropdown(
                label="目前角色",
                choices=list(DEFAULT_PERSONAS.keys()),
                value=initial_sessions[0]["persona"],
                interactive=True
            )
            
            with gr.Accordion("➕ 自訂新角色", open=False):
                p_name = gr.Textbox(label="名稱")
                p_prompt = gr.Textbox(label="指令 (Prompt)", lines=3)
                add_p_btn = gr.Button("新增角色")

        # 主聊天區
        with gr.Column(scale=4, elem_classes="chat-container"):
            gr.Markdown("# 🤖 Gemini 紀錄管理聊天室")
            chatbot = gr.Chatbot(label="Chat Window", height=650, value=initial_sessions[0]["history"])
            
            chat_input = gr.MultimodalTextbox(
                placeholder="輸入訊息或附件...",
                file_count="single",
                file_types=["image", ".pdf"],
                show_label=False
            )

    # 綁定事件
    chat_input.submit(
        chat_response,
        inputs=[chat_input, chatbot, sessions_state, current_session_id, persona_dropdown, personas_state, auto_save_cb],
        outputs=[chatbot, sessions_state, history_list]
    ).then(lambda: gr.update(value=None), outputs=chat_input)

    new_chat_btn.click(
        create_new_chat,
        inputs=[sessions_state, persona_dropdown, personas_state],
        outputs=[chatbot, sessions_state, current_session_id, history_list, persona_dropdown]
    )

    history_list.change(
        switch_chat,
        inputs=[history_list, sessions_state],
        outputs=[chatbot, current_session_id, persona_dropdown]
    )
    
    save_btn.click(manual_save_all, inputs=[sessions_state], outputs=[save_status])
    
    add_p_btn.click(
        lambda n, p, d: (gr.update(choices=list(d.keys()) + [n], value=n), {**d, n: p}, f"✅ 已增：{n}") if n and p else (gr.update(), d, "❌ 資料不全"),
        inputs=[p_name, p_prompt, personas_state],
        outputs=[persona_dropdown, personas_state, save_status]
    )

if __name__ == "__main__":
    demo.launch()
