"""
LangChain + Gemini API 多輪對話互動式聊天程式
支援圖片、PDF 檔案傳送
"""

import sys
import os
import base64
import mimetypes
import json
from datetime import datetime

# 強制 stdout 使用 UTF-8 編碼，避免 Windows 終端編碼問題
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── 1. 載入環境變數 ──────────────────────────────────────
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ 錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案中有設定。")
    exit(1)

# ── 2. 初始化 Gemini 模型 ────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# ── 3. 對話歷史（記憶體） ────────────────────────────────
chat_history = [
    SystemMessage(content="You are a helpful assistant. Please respond in the same language the user uses.")
]

# ── 4. 檔案相關設定 ──────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
PDF_EXTENSION = ".pdf"
ALL_SUPPORTED = IMAGE_EXTENSIONS | {PDF_EXTENSION}

# 專案資料夾路徑（chat.py 所在的目錄）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# CLI 版專屬歷史紀錄資料夾
CLI_HISTORY_DIR = os.path.join(PROJECT_DIR, "cli_history")
os.makedirs(CLI_HISTORY_DIR, exist_ok=True)


def list_project_files():
    """列出專案資料夾中所有支援的檔案"""
    files = []
    for f in os.listdir(PROJECT_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in ALL_SUPPORTED:
            files.append(f)
    return sorted(files)


def find_filename_in_text(text):
    """
    自動偵測訊息中的檔案名稱。
    掃描使用者輸入的每個 token，檢查是否為專案資料夾中存在的檔案。
    回傳 (filename, remaining_text) 或 (None, original_text)
    """
    tokens = text.split()
    for i, token in enumerate(tokens):
        ext = os.path.splitext(token)[1].lower()
        if ext in ALL_SUPPORTED:
            file_path = os.path.join(PROJECT_DIR, token)
            if os.path.isfile(file_path):
                remaining = " ".join(tokens[:i] + tokens[i+1:]).strip()
                return token, remaining
    return None, text


def build_file_message(file_path, user_text):
    """
    根據檔案類型，建立包含文字 + 檔案的多模態訊息。
    """
    ext = os.path.splitext(file_path)[1].lower()

    with open(file_path, "rb") as f:
        file_data = base64.standard_b64encode(f.read()).decode("utf-8")

    if ext in IMAGE_EXTENSIONS:
        mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    elif ext == PDF_EXTENSION:
        mime_type = "application/pdf"
    else:
        return None

    content = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{file_data}"
        },
    })

    return HumanMessage(content=content)

def save_cli_history(json_history):
    """將對話紀錄存為 JSON 檔案"""
    if not json_history:
        return
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp_str}.json"
    filepath = os.path.join(CLI_HISTORY_DIR, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_history, f, ensure_ascii=False, indent=2)
        print(f"\n💾 對話紀錄已儲存至：{filepath}")
    except Exception as e:
        print(f"\n❌ 儲存對話紀錄失敗：{e}")


# ── 5. 終端機互動迴圈 ────────────────────────────────────
def main():
    print("=" * 50)
    print("        🤖 Gemini 聊天機器人")
    print("=" * 50)
    print()
    print("  💬 直接輸入文字即可對話")
    print("  📎 在訊息中加入檔案名稱即可附加檔案")
    print("     例如：photo.jpg 這張圖片是什麼？")
    print("     例如：report.pdf 請摘要這份文件")
    print("  📁 輸入 /files 查看可用檔案")
    print("  🚪 輸入 quit 離開")
    print()

    json_history = []

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 掰掰！")
            save_cli_history(json_history)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("👋 掰掰！下次再聊！")
            save_cli_history(json_history)
            break

        # /files 指令：列出可用檔案
        if user_input.lower() == "/files":
            files = list_project_files()
            if files:
                print("\n📁 專案資料夾中的可用檔案：")
                for f in files:
                    print(f"   • {f}")
            else:
                print("\n📁 專案資料夾中沒有支援的檔案")
                print(f"   支援格式：{', '.join(sorted(ALL_SUPPORTED))}")
            print()
            continue

        # 自動偵測訊息中的檔案名稱
        filename, text = find_filename_in_text(user_input)

        if filename:
            file_path = os.path.join(PROJECT_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()

            if not text:
                text = "請描述這張圖片" if ext in IMAGE_EXTENSIONS else "請摘要這份文件"

            print(f"  📎 偵測到檔案：{filename}")
            message = build_file_message(file_path, text)
        else:
            message = HumanMessage(content=text)

        chat_history.append(message)
        
        # 紀錄 User JSON 格式 (包含時間戳記)
        user_content_str = text if filename else user_input
        json_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": "user",
            "content": f"[附件: {filename}] {user_content_str}" if filename else user_content_str
        })

        try:
            response = llm.invoke(chat_history)
            chat_history.append(AIMessage(content=response.content))
            print(f"\n🤖 Gemini：{response.content}\n")
            
            # 紀錄 AI JSON 格式
            json_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "ai",
                "content": response.content
            })
        except Exception as e:
            print(f"\n❌ 錯誤：{e}\n")
            chat_history.pop()
            json_history.pop() # 若 AI 失敗，也將剛剛加入的 user json 移除


if __name__ == "__main__":
    main()
